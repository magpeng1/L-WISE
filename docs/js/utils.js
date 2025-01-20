/**
 * Utility functions for the image enhancement widget
 */

/**
 * Creates a DOM element with given properties
 * @param {string} tag - HTML tag name
 * @param {Object} props - Properties to set on the element
 * @param {Array} children - Child elements to append
 * @returns {HTMLElement}
 */
export function createElement(tag, props = {}, children = []) {
  const element = document.createElement(tag);
  Object.entries(props).forEach(([key, value]) => {
    if (key === 'className') {
      element.className = value;
    } else if (key === 'style' && typeof value === 'object') {
      Object.assign(element.style, value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      element.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      element.setAttribute(key, value);
    }
  });
  
  children.forEach(child => {
    if (typeof child === 'string') {
      element.appendChild(document.createTextNode(child));
    } else if (child instanceof Node) {
      element.appendChild(child);
    }
  });
  
  return element;
}

/**
 * Creates a debounced version of a function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Milliseconds to wait
 * @returns {Function}
 */
export function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Loads a CSV file and parses it using PapaParse
 * @param {string} url - URL of the CSV file
 * @returns {Promise<Array>}
 */
export async function loadCSV(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    const text = await response.text();
    
    return new Promise((resolve, reject) => {
      Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => resolve(results.data),
        error: (error) => reject(error)
      });
    });
  } catch (error) {
    console.error('Error loading CSV:', error);
    throw error;
  }
}

/**
 * Creates a unique cache key for an image
 * @param {number} index - Image index
 * @param {string|number} epsilon - Epsilon value or 'original'
 * @returns {string}
 */
export function createCacheKey(index, epsilon, dataset) {
  return `${dataset}-${index}-${epsilon}`;
}

/**
 * Formats a class name for display
 * @param {string} rawClass - Raw class name from data
 * @param {string} dataset - Current dataset name
 * @param {Object} mappings - Class name mappings
 * @returns {string}
 */
export function formatClassName(rawClass, dataset, mappings) {
  if (dataset === 'imagenet') {
    return rawClass.replace(/_/g, ' ');
  }
  return mappings[dataset]?.[rawClass] || rawClass;
}

/**
 * Creates a loading spinner element
 * @returns {HTMLElement}
 */
export function createLoadingSpinner() {
  const spinner = createElement('div', { className: 'loading-spinner' });
  spinner.innerHTML = `
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
  `;
  return spinner;
}

/**
 * Creates an error placeholder element
 * @param {string} message - Error message to display
 * @returns {HTMLElement}
 */
export function createErrorPlaceholder(message = 'Failed to load image') {
  return createElement('div', { 
    className: 'error-placeholder' 
  }, [message]);
}

/**
 * Replaces the bucket name in an S3 URL
 * @param {string} url - Original URL
 * @param {string} newBucketName - New bucket name
 * @returns {string}
 */
export function replaceBucketName(url, newBucketName) {
  const urlEnd = url.split('.s3.amazonaws.com')[1];
  return `http://${newBucketName}.s3.amazonaws.com${urlEnd}`;
}

/**
 * Updates a range input's value smoothly
 * @param {HTMLInputElement} input - Range input element
 * @param {number} value - New value
 */
export function updateRangeInput(input, value) {
  input.style.setProperty('--value', value);
  input.value = value;
}
