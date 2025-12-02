import numpy as np
from bytetrack.track_utils.kalman_filter import KalmanFilter
from bytetrack.track_utils.matching import (
    linear_assignment,
    fuse_score,
    euclidean_distance,
    maha_distance,
    iou_distance,
)
from bytetrack.track_utils.basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, point_or_tlwh, score, is_point=False):
        # wait activate
        if is_point:
            # pointの場合、ダミーのtlwhを作成（幅と高さは1.0に設定）
            self._point = np.asarray(point_or_tlwh, dtype=np.float32)
            self._tlwh = np.array(
                [point_or_tlwh[0] - 0.5, point_or_tlwh[1] - 0.5, 1.0, 1.0],
                dtype=np.float32,
            )
        else:
            self._tlwh = np.asarray(point_or_tlwh, dtype=np.float32)
            self._point = self.tlwh_to_xy(self._tlwh)

        # 最新の検出位置を保存
        self._detection_point = self._point.copy()

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            pass  # (x, y)モデルでは不要
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    pass  # (x, y)モデルでは不要
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._point)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # if frame_id == 1:
        #     self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # 最新の検出位置を更新
        self._detection_point = new_track.point.copy()

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.point
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # 最新の検出位置を更新
        self._detection_point = new_track.point.copy()

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.point
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def point(self):
        """Get current position as point (x, y)"""
        if self.mean is None:
            return self._point.copy()
        return self.mean[:2].copy()

    @property
    def detection_point(self):
        """Get the latest detection point (x, y) without Kalman filtering"""
        return self._detection_point.copy()

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        # When Kalman filter state is (x, y, vx, vy)
        # Convert from center coordinates (x, y) to top-left coordinates
        # Keep width and height from original bounding box
        ret = self._tlwh.copy()
        ret[:2] = self.mean[:2] - self._tlwh[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xy(tlwh):
        """Convert bounding box to format `(center x, center y)`.
        旧 tlwh_to_xyah: バウンディングボックスを(center x, center y, aspect ratio, height)に変換
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # 左上座標 + 幅と高さの半分 = 中心座標
        return ret[:2]  # (x, y)のみを返す

    def to_xy(self):
        """旧 to_xyah: 現在のバウンディングボックスを(center x, center y, aspect ratio, height)に変換"""
        return self.tlwh_to_xy(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.match_thresh = args.match_thresh
        self.point_matching = True  # Use point-based matching

        # Distance calculation metric ('maha' or 'euclidean')
        self.distance_metric = args.distance_metric

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 3:  # Point data [x, y, score]
            scores = output_results[:, 2]
            points = output_results[:, :2]  # x, y
        elif output_results.shape[1] == 5:  # Traditional bounding box data
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            # Calculate center point from bounding box
            points = np.column_stack(
                [
                    (bboxes[:, 0] + bboxes[:, 2]) / 2,  # center x
                    (bboxes[:, 1] + bboxes[:, 3]) / 2,  # center y
                ]
            )
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            # Calculate center point from bounding box
            points = np.column_stack(
                [
                    (bboxes[:, 0] + bboxes[:, 2]) / 2,  # center x
                    (bboxes[:, 1] + bboxes[:, 3]) / 2,  # center y
                ]
            )

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        points /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        points_second = points[inds_second]
        points_keep = points[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(points_keep) > 0:
            """Detections"""
            detections = [
                STrack(point, s, is_point=True)
                for (point, s) in zip(points_keep, scores_keep)
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Use distance-based matching
        if self.point_matching:
            # Use selected distance metric of Kalman filter
            dists = maha_distance(
                strack_pool, detections, self.kalman_filter, metric=self.distance_metric
            )
        else:
            # 従来のIoU距離を使用
            dists = iou_distance(strack_pool, detections)

        if not self.args.mot20:
            dists = fuse_score(dists, detections)

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(points_second) > 0:
            """Detections"""
            detections_second = [
                STrack(point, s, is_point=True)
                for (point, s) in zip(points_second, scores_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]

        if self.point_matching:
            # Use distance-based matching
            dists = maha_distance(
                r_tracked_stracks,
                detections_second,
                self.kalman_filter,
                metric=self.distance_metric,
            )
        else:
            # Use traditional IoU distance
            dists = iou_distance(r_tracked_stracks, detections_second)

        matches, u_track, u_detection_second = linear_assignment(
            dists, self.match_thresh
        )
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]

        if self.point_matching:
            # Use distance-based matching (always use Euclidean distance for unconfirmed tracks)
            dists = maha_distance(
                unconfirmed,
                detections,
                self.kalman_filter,
                metric=self.distance_metric,
            )
        else:
            # Use traditional IoU distance
            dists = iou_distance(unconfirmed, detections)

        if not self.args.mot20:
            dists = fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection = linear_assignment(
            dists, self.match_thresh
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            # print("\nnew track")
            # print(track.score)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = euclidean_distance(stracksa, stracksb)

    pairs = np.where(pdist < 10.0)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
