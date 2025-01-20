/**
 * Handles image loading and caching for the widget
 */

import { WIDGET_CONFIG } from './config.js';
import { replaceBucketName } from './utils.js';

export class ImageLoader {
  constructor() {
    this.cache = new Map();
    this.loadingPromises = new Map();
    this.preloadQueue = [];
    this.isProcessingQueue = false;
  }

  /**
   * Creates a unique cache key for an image
   * @param {string} dataset - Dataset name
   * @param {number} index - Image index
   * @param {string|number} epsilon - Epsilon value or 'original'
   * @returns {string}
   */
  createCacheKey(dataset, index, epsilon) {
    return `${dataset}_${index}_${epsilon}`;
  }

  /**
   * Loads a pair of images (original and enhanced) together
   * @param {Object} imageData - Image data object
   * @param {number} index - Image index
   * @param {string} dataset - Dataset name
   * @param {number} epsilon - Epsilon value for enhanced image
   * @param {Function} getBucketName - Function to get bucket name
   * @returns {Promise<{original: HTMLImageElement, enhanced: HTMLImageElement}>}
   */
  async loadImagePair(imageData, index, dataset, epsilon, getBucketName) {
    const originalKey = this.createCacheKey(dataset, index, 'original');
    const enhancedKey = this.createCacheKey(dataset, index, epsilon);

    // Check if both images are already cached
    if (this.cache.has(originalKey) && this.cache.has(enhancedKey)) {
      return {
        original: this.cache.get(originalKey),
        enhanced: this.cache.get(enhancedKey)
      };
    }

    // Check if there's an existing loading promise for this pair
    const loadingKey = `${originalKey}_${enhancedKey}`;
    if (this.loadingPromises.has(loadingKey)) {
      return this.loadingPromises.get(loadingKey);
    }

    // Create new loading promise for the pair
    const loadPromise = new Promise((resolve, reject) => {
      const originalImg = new Image();
      const enhancedImg = new Image();
      let loadedCount = 0;

      const checkComplete = () => {
        if (loadedCount === 2) {
          this.cache.set(originalKey, originalImg);
          this.cache.set(enhancedKey, enhancedImg);
          this.loadingPromises.delete(loadingKey);
          this.maintainCacheSize();
          resolve({ original: originalImg, enhanced: enhancedImg });
        }
      };

      originalImg.onload = () => {
        loadedCount++;
        checkComplete();
      };

      enhancedImg.onload = () => {
        loadedCount++;
        checkComplete();
      };

      const handleError = (error) => {
        this.loadingPromises.delete(loadingKey);
        reject(new Error(`Failed to load image pair: ${error.message}`));
      };

      originalImg.onerror = handleError;
      enhancedImg.onerror = handleError;

      // Start loading both images
      originalImg.src = imageData.url;
      enhancedImg.src = replaceBucketName(imageData.url, getBucketName(epsilon));
    });

    this.loadingPromises.set(loadingKey, loadPromise);
    return loadPromise;
  }

  /**
   * Preloads all epsilon versions for given images
   * @param {Array} imageData - Array of image data objects
   * @param {Array} indices - Array of indices to preload
   * @param {string} dataset - Dataset name
   * @param {Array} epsilons - Array of epsilon values
   * @param {Function} getBucketName - Function to get bucket name
   * @returns {Promise<void>}
   */
  async preloadAllEpsilonVersions(imageData, indices, dataset, epsilons, getBucketName) {
    const promises = [];

    indices.forEach(index => {
      if (index < 0 || index >= imageData.length) return;
      
      const imageInfo = imageData[index];
      if (!imageInfo) return;

      // Load all epsilon versions for each image
      epsilons.forEach(epsilon => {
        promises.push(
          this.loadImagePair(imageInfo, index, dataset, epsilon, getBucketName)
            .catch(error => console.warn(`Failed to preload epsilon ${epsilon} for index ${index}:`, error))
        );
      });
    });

    await Promise.allSettled(promises);
  }

  /**
   * Preloads images for a dataset with all epsilon versions
   * @param {Array} imageData - Array of image data objects
   * @param {string} dataset - Dataset name
   * @param {Array} epsilons - Array of epsilon values
   * @param {Function} getBucketName - Function to get bucket name
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<void>}
   */
  async prefetchDataset(imageData, dataset, epsilons, getBucketName, onProgress) {
    const totalPairs = imageData.length * epsilons.length;
    let loadedPairs = 0;

    const batchSize = 5; // Process in smaller batches to prevent overwhelming the browser
    for (let i = 0; i < imageData.length; i += batchSize) {
      const batch = imageData.slice(i, i + batchSize);
      const promises = batch.map(async (imageInfo, batchIndex) => {
        const index = i + batchIndex;
        try {
          await this.preloadAllEpsilonVersions(
            [imageInfo], 
            [index], 
            dataset, 
            epsilons, 
            getBucketName
          );
          loadedPairs += epsilons.length;
          onProgress(loadedPairs / totalPairs);
        } catch (error) {
          console.error(`Error preloading image batch ${index}:`, error);
        }
      });

      await Promise.allSettled(promises);
      // Small delay between batches to prevent browser lag
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }

  /**
   * Gets all cached versions of an image
   * @param {string} dataset - Dataset name
   * @param {number} index - Image index
   * @returns {Object} Object mapping epsilon values to cached images
   */
  getCachedVersions(dataset, index) {
    const versions = {};
    this.cache.forEach((image, key) => {
      const [cachedDataset, cachedIndex, epsilon] = key.split('_');
      if (cachedDataset === dataset && parseInt(cachedIndex) === index) {
        versions[epsilon] = image;
      }
    });
    return versions;
  }

  /**
   * Maintains cache size within limits
   */
  maintainCacheSize() {
    if (this.cache.size <= WIDGET_CONFIG.CACHE_SIZE * 2) return;

    const entriesToRemove = this.cache.size - (WIDGET_CONFIG.CACHE_SIZE * 2);
    const entries = Array.from(this.cache.entries());
    
    entries
      .slice(0, entriesToRemove)
      .forEach(([key]) => this.cache.delete(key));
  }

  /**
   * Clears all cached images
   */
  clearCache() {
    this.cache.clear();
    this.loadingPromises.clear();
  }
}