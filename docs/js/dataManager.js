/**
 * Manages data loading and state for the widget
 */

import { DATASET_CONFIG } from './config.js';
import { loadCSV } from './utils.js';
import { ImageLoader } from './imageLoader.js';

export class DataManager {
  constructor(onProgress) {
    this.imageLoader = new ImageLoader();
    this.datasets = new Map();
    this.onProgress = onProgress;
    this.currentDataset = null;
  }

  /**
   * Loads all datasets
   * @returns {Promise<void>}
   */
  async loadAllDatasets() {
    const totalDatasets = Object.keys(DATASET_CONFIG).length;
    let loadedDatasets = 0;

    for (const [datasetName, config] of Object.entries(DATASET_CONFIG)) {
      try {
        const csvPath = `assets/${config.filename}`;
        const data = await loadCSV(csvPath);
        const shuffledData = this.shuffleData(data);
        this.datasets.set(datasetName, shuffledData);
        
        // Only preload first 20 images initially
        const initialBatch = shuffledData.slice(0, 20);
        await this.imageLoader.prefetchDataset(
          initialBatch,
          datasetName,
          config.epsilons,
          config.getBucketName,
          (progress) => {
            const totalProgress = (loadedDatasets + progress) / totalDatasets;
            this.onProgress(totalProgress);
          }
        );

        loadedDatasets++;
        
        // Start loading remaining images in the background
        this.loadRemainingImages(datasetName, shuffledData.slice(20));
      } catch (error) {
        console.error(`Error loading dataset ${datasetName}:`, error);
        throw error;
      }
    }
  }

  /**
   * Shuffles array using Fisher-Yates algorithm
   * @param {Array} array - Array to shuffle
   * @returns {Array}
   */
  shuffleData(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  /**
   * Gets data for a specific dataset
   * @param {string} dataset - Dataset name
   * @returns {Array}
   */
  getDataset(dataset) {
    return this.datasets.get(dataset) || [];
  }

  /**
   * Gets configuration for a dataset
   * @param {string} dataset - Dataset name
   * @returns {Object}
   */
  getDatasetConfig(dataset) {
    return DATASET_CONFIG[dataset];
  }

  /**
   * Preloads images for upcoming indices
   * @param {string} dataset - Dataset name
   * @param {number} currentIndex - Current image index
   * @param {number} count - Number of images to preload
   * @returns {Promise<void>}
   */
  async preloadUpcoming(dataset, currentIndex, count) {
    const data = this.getDataset(dataset);
    const config = this.getDatasetConfig(dataset);
    const indices = [];

    for (let i = 1; i <= count; i++) {
      indices.push((currentIndex + i) % data.length);
    }

    await this.imageLoader.preloadAllEpsilonVersions(
      data,
      indices,
      dataset,
      config.epsilons,
      config.getBucketName
    );
  }

  /**
   * Gets visible images for current view
   * @param {string} dataset - Dataset name
   * @param {number} currentIndex - Current image index
   * @param {number} visibleCount - Number of visible images
   * @returns {Array}
   */
  getVisibleImages(dataset, currentIndex, visibleCount) {
    const data = this.getDataset(dataset);
    const images = [];
    
    for (let i = 0; i < visibleCount; i++) {
      const index = (currentIndex + i) % data.length;
      images.push({
        ...data[index],
        index
      });
    }

    return images;
  }

  /**
   * Loads remaining images in the background
   * @param {string} datasetName - Dataset name
   * @param {Array} remainingData - Array of remaining image data
   * @returns {Promise<void>}
   */
  async loadRemainingImages(datasetName, remainingData) {
    const config = this.getDatasetConfig(datasetName);
    const batchSize = 20;
    
    for (let i = 0; i < remainingData.length; i += batchSize) {
      const batch = remainingData.slice(i, i + batchSize);
      await this.imageLoader.prefetchDataset(
        batch,
        datasetName,
        config.epsilons,
        config.getBucketName,
        () => {} // No progress callback needed for background loading
      );
      
      // Small delay to prevent overwhelming the browser
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  /**
   * Cleans up resources
   */
  cleanup() {
    this.imageLoader.clearCache();
    this.datasets.clear();
  }
}