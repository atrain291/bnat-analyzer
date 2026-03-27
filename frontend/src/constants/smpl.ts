export const SMPL_JOINT_NAMES = [
  "pelvis", "left_hip", "right_hip", "spine1",
  "left_knee", "right_knee", "spine2",
  "left_ankle", "right_ankle", "spine3",
  "left_foot", "right_foot", "neck",
  "left_collar", "right_collar", "head",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_hand", "right_hand",
] as const;

export const SMPL_BONE_CONNECTIONS: [number, number][] = [
  // Spine chain
  [0, 3],   // pelvis → spine1
  [3, 6],   // spine1 → spine2
  [6, 9],   // spine2 → spine3
  [9, 12],  // spine3 → neck
  [12, 15], // neck → head

  // Left arm
  [9, 13],  // spine3 → left_collar
  [13, 16], // left_collar → left_shoulder
  [16, 18], // left_shoulder → left_elbow
  [18, 20], // left_elbow → left_wrist
  [20, 22], // left_wrist → left_hand

  // Right arm
  [9, 14],  // spine3 → right_collar
  [14, 17], // right_collar → right_shoulder
  [17, 19], // right_shoulder → right_elbow
  [19, 21], // right_elbow → right_wrist
  [21, 23], // right_wrist → right_hand

  // Left leg
  [0, 1],   // pelvis → left_hip
  [1, 4],   // left_hip → left_knee
  [4, 7],   // left_ankle → left_ankle
  [7, 10],  // left_ankle → left_foot

  // Right leg
  [0, 2],   // pelvis → right_hip
  [2, 5],   // right_hip → right_knee
  [5, 8],   // right_knee → right_ankle
  [8, 11],  // right_ankle → right_foot
];

export const FOOT_JOINT_INDICES = [7, 8, 10, 11] as const; // left_ankle, right_ankle, left_foot, right_foot
