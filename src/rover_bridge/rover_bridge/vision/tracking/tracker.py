"""
Tracker — scores every detected person each frame and picks the best match.
Outputs a TrackResult with confidence score and lock_mode.

This is the brain of Phase 1.
Extend this class in Phase 2 to also handle room-exit logic.
"""
import numpy as np
import cv2
from rover_bridge.vision.core.data_types import Detection, FaceDetection, TrackResult, BoundingBox
from rover_bridge.vision.tracking.target_profile import TargetProfile, _color_histogram
from rover_bridge.vision.perception.face_recognizer import FaceRecognizer
from rover_bridge.vision.utils.logger import RoverLogger


class Tracker:
    def __init__(self, settings, face_recognizer = None):
        self.settings        = settings
        self.logger          = RoverLogger("Tracker")
        self.face_recognizer = face_recognizer
        self._last_bbox      = None
        self._frame_num:     int = 0
        self._lost_streak    = 0
        self._smoothed_offset_x = None
        self._smoothed_area_ratio = None
        self._last_result = None
        self.LOST_STREAK_LIMIT = 2
        self._calib_areas = []

    def update(self, frame, detections: list[Detection], faces: list[FaceDetection], profile: TargetProfile) -> TrackResult:
        self._frame_num += 1
        h, w = frame.shape[:2]

        #if self._frame_num % self.settings.FACE_RECOGNITION_INTERVAL == 0:
        #    self._cached_faces = self.face_recognizer.detect(frame)
        #faces = self._cached_faces

        persons = [d for d in detections if d.class_id == 0]
        if not persons:
            self._lost_streak += 1
            if self._lost_streak >= self.LOST_STREAK_LIMIT or self._last_result is None:
                return self._lost_result()
            return self._last_result

        best_score, best_person, best_mode = 0.0, None, "lost"

        for person in persons:
            score, mode = self._score_person(frame, person, faces, profile)
            if score > best_score:
                best_score, best_person, best_mode = score, person, mode

        if best_person is None or best_score < self.settings.RECALIBRATE_THRESHOLD:
            self._lost_streak += 1 # scoring
            if self._lost_streak >= self.LOST_STREAK_LIMIT or self._last_result is None:
                return self._lost_result()

            return self._last_result

        fixed_ref = getattr(self.settings, "FIXED_REFERENCE_BBOX_AREA", 0)
        if fixed_ref and profile.reference_bbox_area in (0, None):
            profile.reference_bbox_area = fixed_ref
            self.logger.info(f"Using fixed reference_bbox_area={fixed_ref}px^2")

        if profile.reference_bbox_area in (0, None):
            b = best_person.bbox
            m = 8   # px margin from frame edge
            off = abs((b.center_x - w / 2) / (w / 2))
            cropped = b.x1 <= m or b.x2 >= w - m or b.y1 <= m

            if cropped or off > 0.25:
                self._calib_areas = []          # partial / off-center: start over
                self.logger.info(f"Calib wait: cropped={cropped} off={off:.2f} "
                                 f"box=({b.x1},{b.y1},{b.x2},{b.y2})")
                return self._lost_result()

            self._calib_areas.append(b.area)
            self._calib_areas = self._calib_areas[-3:]
            if len(self._calib_areas) < 3 or max(self._calib_areas) > 1.25 * min(self._calib_areas):
                self.logger.info(f"Calib wait: steady {len(self._calib_areas)}/3 {self._calib_areas}")
                return self._lost_result()

            profile.reference_bbox_area = sorted(self._calib_areas)[1]   # median of 3
            self._calib_areas = []
            self.logger.info(
                f"Auto-calibrated reference_bbox_area={profile.reference_bbox_area}px^2 "
                f"from 3 steady frames ({best_mode}).")

        # if best_mode == "face" and profile.reference_bbox_area in (0, None):
        #    profile.reference_bbox_area = best_person.bbox.area
        #    self.logger.info(f"Auto-calibrated reference_bbox_area={best_person.bbox.area}px^2 from first face match.")

        self._lost_streak = 0
        self._last_bbox = best_person.bbox
        bbox            = best_person.bbox
        offset_x        = (bbox.center_x - w / 2) / (w / 2)      # -1 to +1
        offset_y        = (bbox.center_y - h/ 2) / (h / 2)
        area_ratio      = bbox.area / max(profile.reference_bbox_area, 1)

        alpha = 0.3
        if self._smoothed_offset_x is None:
            self._smoothed_offset_x = offset_x
            self._smoothed_area_ratio = area_ratio
        else:
            self._smoothed_offset_x = alpha * offset_x + (1 - alpha) * self._smoothed_offset_x
            self._smoothed_area_ratio = alpha * area_ratio + (1 - alpha) * self._smoothed_area_ratio

        offset_x = self._smoothed_offset_x
        area_ratio = self._smoothed_area_ratio

        result = TrackResult(
            target_bbox    = bbox,
            confidence     = best_score,
            lock_mode      = best_mode,
            offset_x       = offset_x,
            offset_y       = offset_y,
            bbox_area_ratio= area_ratio,
            frame_number   = self._frame_num,
            lost_streak    = self._lost_streak
        )
        self._last_result = result
        return result

    def _score_person(self, frame, person: Detection, faces: list[FaceDetection], profile: TargetProfile):
        s    = self.settings
        bbox = person.bbox
        score, mode = 0.0, "body"

        # ── Face match ────────────────────────────────────────
        if profile.face_embedding is not None and faces and self.face_recognizer is not None:
            for face in faces:
                if self._bbox_overlaps(face.bbox, bbox):
                    face_score = self.face_recognizer.match(face.embedding, profile.face_embedding)
                    if face_score > 0:
                        score += face_score * s.WEIGHT_FACE
                        mode   = "face"

        # ── Color histogram match ─────────────────────────────
        if profile.upper_body_histogram is not None:
            upper_hist = _color_histogram(frame, bbox, "upper")
            color_sim  = cv2.compareHist(
                profile.upper_body_histogram.reshape(-1, 1).astype(np.float32),
                upper_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            score += max(0, color_sim) * s.WEIGHT_COLOR

        # ── Visibility mode ───────────────────────────────────
        h = frame.shape[0]
        visible_fraction = (bbox.y2 - bbox.y1) / h
        if visible_fraction > 0.5:
            score += s.WEIGHT_BODY
            if mode != "face": mode = "body"
        elif visible_fraction > 0.25:
            score += s.WEIGHT_LOWER_BODY
            if mode not in ("face",): mode = "lower_body"
        else:
            score += s.WEIGHT_FEET
            if mode not in ("face", "body"): mode = "feet"

        # ── Continuity bonus ──────────────────────────────────
        if self._last_bbox:
            dist = abs(bbox.center_x - self._last_bbox.center_x) + abs(bbox.center_y - self._last_bbox.center_y)
            if dist < 80:
                score += s.WEIGHT_CONTINUITY

        return min(score, 1.0), mode

    def _bbox_overlaps(self, a: BoundingBox, b: BoundingBox) -> bool:
        return not (a.x2 < b.x1 or a.x1 > b.x2 or a.y2 < b.y1 or a.y1 > b.y2)

    def _lost_result(self):
        self._smoothed_offset_x = None
        self._smoothed_area_ratio = None
        return TrackResult(
            target_bbox=None,
            confidence=0.0,
            lock_mode="lost",
            offset_x=0.0,
            bbox_area_ratio=0.0,
            frame_number=self._frame_num,
            lost_streak = self._lost_streak
        )
