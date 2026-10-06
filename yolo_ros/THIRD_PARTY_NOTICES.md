# Third-party notices

`yolo_ros` is distributed under the MIT License. Portions are based on the
following permissively licensed projects. These notices are retained in
addition to `LICENSE`.

## ByteTrack

The tracking implementation in `src/tracking/` and `include/yolo_ros/tracking/`
is based on the ByteTrack paper and the original ByteTrack reference code,
including its Python tracker and C++ deployment implementations:

- Project: <https://github.com/ifzhang/ByteTrack>
- Paper: <https://arxiv.org/abs/2110.06864>
- License: MIT

The ROS-facing detection metadata fields and parameter names (for example,
`track_high_thresh`, `track_low_thresh`, class IDs, and input detection
indices) retain Ultralytics-compatible interface conventions. The tracking
algorithm, lifecycle, Kalman model, and association helpers use the original
MIT-licensed ByteTrack code as their licensing basis. LAPJV has the additional
BSD-2-Clause notice below.

MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BoT-SORT

The BoT-SORT tracker (`src/tracking/bot_sort.cpp`,
`src/tracking/utils/camera_motion.cpp` and their headers) is based on the
BoT-SORT paper and reference implementation:

- Project: <https://github.com/NirAharon/BoT-SORT>
- Paper: <https://arxiv.org/abs/2206.14651>
- License: MIT

The XYWH Kalman filter, the camera-motion compensator and the association
pipeline are ported from the reference `tracker/kalman_filter.py`,
`tracker/gmc.py` and `tracker/bot_sort.py`. The ReID/appearance branch IS
included as the opt-in `with_reid` appearance association, using embeddings from
a pluggable ONNX encoder (see the torchreid/OSNet and FastReID sections below).
The reference is itself based on the MIT-licensed
ByteTrack reference (see above).

MIT License

Copyright (c) 2022 Nir Aharon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## YOLOs-CPP

Portions of the ONNX Runtime inference and YOLO detection/segmentation code were
adapted from this historical YOLOs-CPP snapshot:

- Project: <https://github.com/Geekgineer/YOLOs-CPP>
- Snapshot: `6faab6e10244edf6a2d9f479d901d0b7405055ec` (2025-02-21)
- Snapshot license declaration: MIT
- Declaration: <https://github.com/Geekgineer/YOLOs-CPP/blob/6faab6e10244edf6a2d9f479d901d0b7405055ec/README.md#license>

That snapshot's README stated: "This project is licensed under the MIT License."
It did not contain a standalone `LICENSE` file or a separate copyright notice.
This notice preserves the project attribution and the exact source snapshot.
Later YOLOs-CPP revisions and their current license are not the licensing basis
for these portions.

## lap

The ByteTrack C++ deployment's LAPJV implementation, adapted in
`src/tracking/utils/lapjv.cpp`, incorporates the `lap` project's dense LAPJV
solver:

- Project: <https://github.com/gatagat/lap>
- License: BSD-2-Clause

BSD 2-Clause License

Copyright (c) 2012-2025, Tomas Kazmar

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## yolo_ros ports

The C++ ROS node implementations adapt behavior from the original `yolo_ros`
Python nodes. Miguel Ángel González Santamarta, the copyright holder of those
contributions, authorized their release in the MIT-licensed C++ pipeline.

## torchreid / deep-person-reid (OSNet ReID backbone)

The optional BoT-SORT-ReID appearance branch
(`src/engine/reid_encoder.cpp`, the feature code in `src/tracking/strack.cpp`
and the appearance fusion in `src/tracking/bot_sort.cpp`) supports a ReID
embedding model exported from torchreid's OSNet:

- Project: <https://github.com/KaiyangZhou/deep-person-reid>
- Paper: <https://arxiv.org/abs/1905.00953>
- License: MIT (OSNet architecture, code and the model-zoo weights)

The C++ code does not embed torchreid source; it only defines the ONNX input
contract documented in `docs/models.md`.

## FastReID (alternative ReID backbone)

The same appearance branch also supports FastReID's SBS-S50 model, optionally
using the ReID weights released by BoT-SORT (MIT, see the BoT-SORT notice):

- Project: <https://github.com/JDAI-CV/fast-reid>
- Paper: <https://arxiv.org/abs/2006.02631>
- License: Apache-2.0 (code)

No FastReID source is bundled; only the exported ONNX model is consumed.
