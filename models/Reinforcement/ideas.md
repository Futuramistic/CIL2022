Basic Ideas:
- use reinforcement learning as a refinement tool after segmentation --> input is raw image and segmentation
- only use reinforcement learning to segment the raw image --> input is raw image and segmentation so far?
- state space is all possibilities every pixel can have
- action and transition is both changing a pixel state (foreground/background), network sees state of complete segmented image at all times
or
- network only has local information and starts at a point: action is painting road pixels within that patch (velocity, angle, put down brush)
- penalize: putting a brush down, changing the road width within the same stroke, wrong segmentation