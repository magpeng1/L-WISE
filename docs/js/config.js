/**
 * Configuration constants for the image enhancement widget
 */

export const DATASET_CONFIG = {
  'imagenet': {
    filename: 'imagenet16_dirmap.csv',
    defaultEpsilon: 20,
    epsilons: [5, 10, 15, 20],
    getBucketName: (eps) => `morgan-imagenet16-${eps}-0dot5-${eps*2}-logit`,
  },
  'moths': {
    filename: 'idaea4_dirmap.csv',
    defaultEpsilon: 8,
    epsilons: [0.25, 0.5, 1, 2, 4, 8],
    getBucketName: (eps) => `morgan-idaea4-${eps.toString().replace('.', 'dot')}-0dot5-${Math.max(1, eps * 2)}-logit-diverge`,
  },
  'dermoscopy': {
    filename: 'ham4_dirmap.csv',
    defaultEpsilon: 8,
    epsilons: [0.25, 0.5, 1, 2, 4, 8],
    getBucketName: (eps) => `morgan-ham10000-${eps.toString().replace('.', 'dot')}-0dot5-${Math.max(1, eps * 2)}-logit-diverge`,
  },
  'histology': {
    filename: 'mhist_dirmap.csv',
    defaultEpsilon: 8,
    epsilons: [0.25, 0.5, 1, 2, 4, 8],
    getBucketName: (eps) => `morgan-mhist-full-${eps.toString().replace('.', 'dot')}-0dot5-${Math.max(1, eps * 2)}-logit-diverge`,
  }
};

export const CLASS_MAPPINGS = {
  'moths': {
    "01233_Animalia_Arthropoda_Insecta_Lepidoptera_Geometridae_Idaea_aversata": "aversata",
    "01234_Animalia_Arthropoda_Insecta_Lepidoptera_Geometridae_Idaea_biselata": "biselata",
    "01239_Animalia_Arthropoda_Insecta_Lepidoptera_Geometridae_Idaea_seriata": "seriata",
    "01240_Animalia_Arthropoda_Insecta_Lepidoptera_Geometridae_Idaea_tacturata": "tacturata"
  },
  'dermoscopy': {
    "nv": "benign\nmole",
    "mel": " melanoma",
    "bkl": "benign\nkeratosis",
    "bcc": "basal cell\ncarcinoma"
  },
  'histology': {
    "hp": "hyperplastic\npolyp",
    "ssa": "sessile serrated\nadenoma"
  }
};

export const WIDGET_CONFIG = {
  VISIBLE_IMAGES: 5,
  SCROLL_INTERVAL: 3000,
  DATASET_SWITCH_INTERVAL: 9000,
  IMAGE_WIDTH: 120,
  IMAGE_HEIGHT: 120,
  TRANSITION_DURATION: 100,
  CACHE_SIZE: 50, // Maximum number of images to keep in memory
};

export const ANIMATION_CONFIG = {
  EASING: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t, // Smooth easing function
  FRAME_RATE: 60,
};