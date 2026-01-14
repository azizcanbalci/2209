"""
MOD 7: Stable Visual SLAM with Good Features + Optical Flow
- More stable feature tracking
- Consistent map point management  
- Proper trajectory visualization
"""

import cv2
import numpy as np
import threading
import os


class MapPoint:
    """3D Map Point with stability tracking"""
    __slots__ = ['id', 'pos', 'descriptor', 'observations', 'last_seen', 'bad', 'color', 'quality']
    _counter = 0
    
    def __init__(self, pos, desc=None, color=None):
        self.id = MapPoint._counter
        MapPoint._counter += 1
        self.pos = pos.astype(np.float32).copy() if hasattr(pos, 'astype') else np.array(pos, dtype=np.float32)
        self.descriptor = desc
        self.observations = 1
        self.last_seen = 0
        self.bad = False
        self.color = color if color is not None else (0, 0, 0)
        self.quality = 1.0


class VisualSLAM:
    def __init__(self, camera_matrix=None):
        print("🗺️ Stable Visual SLAM starting...")
        
        # Camera intrinsics (typical webcam)
        if camera_matrix is None:
            self.K = np.array([
                [500.0, 0, 320.0],
                [0, 500.0, 240.0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.K = camera_matrix.astype(np.float32)
        
        self.fx, self.fy = self.K[0,0], self.K[1,1]
        self.cx, self.cy = self.K[0,2], self.K[1,2]
        
        # Good Features to Track - more stable than ORB
        self.feature_params = dict(
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7,
            useHarrisDetector=True,
            k=0.04
        )
        
        # Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # State
        self.state = "INIT"
        self.frame_id = 0
        
        # Previous frame data
        self.prev_gray = None
        self.prev_pts = None
        
        # Pose tracking
        self.R_total = np.eye(3, dtype=np.float32)
        self.t_total = np.zeros((3, 1), dtype=np.float32)
        
        # Map data
        self.map_points = []
        self.keyframes = []
        self.trajectory = [(0.0, 0.0, 0.0)]
        
        # Visualization
        self.tracked_pts_2d = []
        self.match_count = 0
        
        # Keyframe parameters
        self.last_kf_pos = np.zeros(3)
        self.last_kf_frame = 0
        self.kf_distance = 0.15  # 15cm for new keyframe
        self.kf_frames = 25
        
        self._lock = threading.Lock()
        print("✓ SLAM ready!")
    
    def process_frame(self, frame):
        """Process a new frame"""
        with self._lock:
            self.frame_id += 1
            
            # Resize
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, 480))
            
            # Grayscale + enhance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            gray = cv2.equalizeHist(gray)
            
            if self.state == "INIT":
                return self._initialize(gray)
            else:
                return self._track(gray)
    
    def _detect_features(self, gray):
        """Detect good features"""
        pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        if pts is not None:
            return pts.reshape(-1, 2)
        return np.array([])
    
    def _initialize(self, gray):
        """Initialize with first frame"""
        pts = self._detect_features(gray)
        
        if len(pts) < 100:
            return False
        
        self.prev_gray = gray.copy()
        self.prev_pts = pts
        self.tracked_pts_2d = pts.astype(np.int32).tolist()
        
        # First keyframe
        self.keyframes.append({
            'frame_id': self.frame_id,
            'R': self.R_total.copy(),
            't': self.t_total.copy(),
            'pts': pts.copy()
        })
        self.last_kf_frame = self.frame_id
        
        self.state = "OK"
        print(f"✓ Initialized with {len(pts)} features")
        return True
    
    def _track(self, gray):
        """Track features with optical flow"""
        if self.prev_pts is None or len(self.prev_pts) < 20:
            self.prev_pts = self._detect_features(self.prev_gray if self.prev_gray is not None else gray)
            if len(self.prev_pts) < 50:
                self.state = "LOST"
                return False
        
        # Optical flow tracking
        prev_pts = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.state = "LOST"
            return False
        
        # Filter good tracks
        status = status.ravel()
        good_prev = prev_pts[status == 1].reshape(-1, 2)
        good_curr = curr_pts[status == 1].reshape(-1, 2)
        
        # Filter by error
        if err is not None:
            err = err.ravel()[status == 1]
            mask = err < 12
            good_prev = good_prev[mask]
            good_curr = good_curr[mask]
        
        if len(good_curr) < 20:
            self.prev_pts = self._detect_features(gray)
            self.prev_gray = gray.copy()
            return len(self.prev_pts) > 50
        
        # Essential matrix
        E, mask = cv2.findEssentialMat(
            good_prev, good_curr, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is not None and mask is not None:
            inliers = mask.ravel() == 1
            good_prev_in = good_prev[inliers]
            good_curr_in = good_curr[inliers]
            
            if len(good_prev_in) > 10:
                _, R, t, _ = cv2.recoverPose(E, good_prev_in, good_curr_in, self.K)
                
                if R is not None and t is not None:
                    # Scale from flow magnitude
                    flow = good_curr_in - good_prev_in
                    median_flow = np.median(np.linalg.norm(flow, axis=1))
                    motion_scale = min(0.03, median_flow * 0.0015)
                    
                    # Update pose
                    t_scaled = t * motion_scale
                    self.t_total = self.t_total + self.R_total @ t_scaled
                    self.R_total = R @ self.R_total
        
        # Camera position
        cam_pos = -self.R_total.T @ self.t_total
        pos = (float(cam_pos[0,0]), float(cam_pos[1,0]), float(cam_pos[2,0]))
        self.trajectory.append(pos)
        
        # Store for visualization
        self.tracked_pts_2d = good_curr.astype(np.int32).tolist()
        self.match_count = len(good_curr)
        
        # Keyframe check
        if self._should_create_keyframe(pos):
            self._create_keyframe(gray, good_curr, good_prev, pos)
        
        # Refresh features if low
        if len(good_curr) < 150:
            new_pts = self._detect_features(gray)
            if len(new_pts) > 0:
                good_curr = np.vstack([good_curr, new_pts[:400 - len(good_curr)]])
        
        self.prev_pts = good_curr[:400]
        self.prev_gray = gray.copy()
        self.state = "OK"
        
        return True
    
    def _should_create_keyframe(self, pos):
        """Check if new keyframe needed"""
        dist = np.linalg.norm(np.array(pos) - self.last_kf_pos)
        if dist > self.kf_distance:
            return True
        if self.frame_id - self.last_kf_frame > self.kf_frames:
            return True
        return False
    
    def _create_keyframe(self, gray, curr_pts, prev_pts, pos):
        """Create keyframe and triangulate points"""
        if len(self.keyframes) < 1:
            return
        
        prev_kf = self.keyframes[-1]
        
        kf = {
            'frame_id': self.frame_id,
            'R': self.R_total.copy(),
            't': self.t_total.copy(),
            'pts': curr_pts.copy()
        }
        self.keyframes.append(kf)
        self.last_kf_frame = self.frame_id
        self.last_kf_pos = np.array(pos)
        
        # Triangulate
        if len(curr_pts) > 15 and len(prev_pts) > 15:
            min_pts = min(len(curr_pts), len(prev_pts))
            self._triangulate_points(prev_kf, kf, prev_pts[:min_pts], curr_pts[:min_pts])
        
        # Limit keyframes
        if len(self.keyframes) > 50:
            self.keyframes = self.keyframes[-50:]
    
    def _triangulate_points(self, kf1, kf2, pts1, pts2):
        """Triangulate 3D points"""
        P1 = self.K @ np.hstack([kf1['R'], kf1['t']])
        P2 = self.K @ np.hstack([kf2['R'], kf2['t']])
        
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
        pts3d = (pts4d[:3] / pts4d[3]).T
        
        new_count = 0
        for i, pt in enumerate(pts3d):
            if not np.isfinite(pt).all():
                continue
            
            # Check depth
            pt_cam = kf2['R'] @ pt.reshape(3,1) + kf2['t']
            if pt_cam[2,0] < 0.1:
                continue
            
            # Distance filter
            dist = np.linalg.norm(pt)
            if dist < 0.1 or dist > 30:
                continue
            
            mp = MapPoint(pt.astype(np.float32))
            mp.observations = 2
            mp.last_seen = self.frame_id
            self.map_points.append(mp)
            new_count += 1
        
        # Limit map points
        if len(self.map_points) > 2500:
            self.map_points = self.map_points[-2000:]
        
        if new_count > 0:
            print(f"  +{new_count} pts (total: {len(self.map_points)})")
    
    def get_visualization(self, frame):
        """Draw tracked features on frame"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Draw tracked points as green circles
        for pt in self.tracked_pts_2d[:300]:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        
        # Status bar
        color = (0, 255, 0) if self.state == "OK" else (0, 165, 255)
        cv2.rectangle(vis, (0, h-30), (w, h), (0, 0, 0), -1)
        
        status = f"{self.state} | Features: {self.match_count} | MPs: {len(self.map_points)} | KFs: {len(self.keyframes)}"
        cv2.putText(vis, status, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis
    
    def get_topdown_map(self, size=600):
        """
        ORB-SLAM3 Style Top-down Map
        - White background
        - Black dots for map points
        - Blue squares + line for keyframes
        - Red dot for current position (MOVES!)
        """
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Need at least some data
        if len(self.trajectory) < 2:
            cv2.putText(img, "Move camera to start...", (size//4, size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return img
        
        # Collect all points for auto-scaling
        all_x, all_z = [], []
        
        # Add trajectory points
        for p in self.trajectory:
            all_x.append(p[0])
            all_z.append(p[2])
        
        # Add map points
        for mp in self.map_points:
            if not mp.bad:
                all_x.append(mp.pos[0])
                all_z.append(mp.pos[2])
        
        if len(all_x) < 2:
            return img
        
        # Calculate bounds and scale
        min_x, max_x = min(all_x), max(all_x)
        min_z, max_z = min(all_z), max(all_z)
        
        range_x = max(max_x - min_x, 0.5)
        range_z = max(max_z - min_z, 0.5)
        max_range = max(range_x, range_z) * 1.4
        
        scale = (size - 100) / max_range
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        def to_pixel(x, z):
            px = int(size//2 + (x - center_x) * scale)
            py = int(size//2 - (z - center_z) * scale)
            return px, py
        
        # 1. Draw map points (small black dots)
        for mp in self.map_points:
            if mp.bad:
                continue
            px, py = to_pixel(mp.pos[0], mp.pos[2])
            if 5 <= px < size-5 and 5 <= py < size-5:
                cv2.circle(img, (px, py), 2, (50, 50, 50), -1)
        
        # 2. Draw trajectory (light blue line)
        traj_pixels = []
        for p in self.trajectory:
            px, py = to_pixel(p[0], p[2])
            if 0 <= px < size and 0 <= py < size:
                traj_pixels.append((px, py))
        
        if len(traj_pixels) > 1:
            for i in range(1, len(traj_pixels)):
                cv2.line(img, traj_pixels[i-1], traj_pixels[i], (255, 200, 100), 2)
        
        # 3. Draw keyframe positions (blue squares)
        for kf in self.keyframes:
            pos = -kf['R'].T @ kf['t']
            px, py = to_pixel(pos[0,0], pos[2,0])
            if 5 <= px < size-5 and 5 <= py < size-5:
                cv2.rectangle(img, (px-4, py-4), (px+4, py+4), (255, 0, 0), 2)
        
        # 4. Draw start position (green circle)
        if len(traj_pixels) > 0:
            cv2.circle(img, traj_pixels[0], 8, (0, 200, 0), -1)
        
        # 5. Draw CURRENT position (RED - this MOVES!)
        if len(self.trajectory) > 0:
            curr_pos = self.trajectory[-1]
            px, py = to_pixel(curr_pos[0], curr_pos[2])
            if 5 <= px < size-5 and 5 <= py < size-5:
                cv2.circle(img, (px, py), 12, (0, 0, 255), -1)
                cv2.circle(img, (px, py), 14, (255, 255, 255), 2)
        
        # Info text
        cv2.putText(img, f"KeyFrames: {len(self.keyframes)}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"MapPoints: {len(self.map_points)}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"Trajectory: {len(self.trajectory)}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Legend
        ly = size - 50
        cv2.circle(img, (size-80, ly), 4, (50, 50, 50), -1)
        cv2.putText(img, "Map", (size-65, ly+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        cv2.rectangle(img, (size-84, ly+12), (size-76, ly+20), (255, 0, 0), 2)
        cv2.putText(img, "KF", (size-65, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        cv2.circle(img, (size-80, ly+32), 6, (0, 0, 255), -1)
        cv2.putText(img, "Pos", (size-65, ly+36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        
        return img
    
    def get_stats(self):
        """Get current stats"""
        pos = self.trajectory[-1] if self.trajectory else (0, 0, 0)
        return {
            'frame': self.frame_id,
            'state': self.state,
            'kfs': len(self.keyframes),
            'mps': len(self.map_points),
            'tracked': self.match_count,
            'pos': pos
        }
    
    def reset(self):
        """Reset SLAM"""
        with self._lock:
            self.state = "INIT"
            self.frame_id = 0
            self.prev_gray = None
            self.prev_pts = None
            self.R_total = np.eye(3, dtype=np.float32)
            self.t_total = np.zeros((3, 1), dtype=np.float32)
            self.map_points = []
            self.keyframes = []
            self.trajectory = [(0.0, 0.0, 0.0)]
            self.tracked_pts_2d = []
            self.match_count = 0
            self.last_kf_pos = np.zeros(3)
            self.last_kf_frame = 0
            MapPoint._counter = 0
        print("🔄 SLAM reset")
    
    def save_map(self, path):
        """Save map to PLY file"""
        if len(self.map_points) < 10:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(self.map_points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("end_header\n")
                for mp in self.map_points:
                    f.write(f"{mp.pos[0]:.4f} {mp.pos[1]:.4f} {mp.pos[2]:.4f}\n")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def load_map(self, path):
        """Load map from PLY file"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            start = 0
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    start = i + 1
                    break
            self.map_points = []
            for line in lines[start:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
                    self.map_points.append(MapPoint(pos))
            print(f"Loaded {len(self.map_points)} points")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False


# Global instance
_slam = None

def get_slam():
    global _slam
    if _slam is None:
        _slam = VisualSLAM()
    return _slam

def process_frame(frame):
    return get_slam().process_frame(frame)

def get_visualization(frame):
    return get_slam().get_visualization(frame)

def get_topdown_map(size=600):
    return get_slam().get_topdown_map(size)

def get_stats():
    return get_slam().get_stats()

def reset():
    get_slam().reset()

def save_map(path):
    return get_slam().save_map(path)

def load_map(path):
    return get_slam().load_map(path)

def init(cam=None):
    """Initialize SLAM (for compatibility)"""
    global _slam
    _slam = VisualSLAM(cam)
    return True
