/**
 * Handles smooth animations using requestAnimationFrame
 */

import { ANIMATION_CONFIG, WIDGET_CONFIG } from './config.js';

export class AnimationManager {
  constructor() {
    this.animations = new Map();
    this.frameId = null;
  }

  /**
   * Starts a new animation
   * @param {string} id - Unique animation identifier
   * @param {Object} options - Animation options
   * @param {number} options.duration - Animation duration in ms
   * @param {number} options.startValue - Starting value
   * @param {number} options.endValue - Ending value
   * @param {Function} options.onUpdate - Update callback
   * @param {Function} options.onComplete - Completion callback
   * @returns {Promise<void>}
   */
  animate(id, { duration, startValue, endValue, onUpdate, onComplete }) {
    // Cancel any existing animation with this ID
    this.cancel(id);

    return new Promise((resolve) => {
      const startTime = performance.now();
      const animationState = {
        startTime,
        duration,
        startValue,
        endValue,
        onUpdate,
        resolve
      };

      this.animations.set(id, animationState);
      
      if (!this.frameId) {
        this.frameId = requestAnimationFrame(this.update.bind(this));
      }
    }).then(() => {
      if (onComplete) onComplete();
    });
  }

  /**
   * Animation frame update function
   * @param {number} currentTime - Current timestamp
   */
  update(currentTime) {
    let hasActiveAnimations = false;

    this.animations.forEach((state, id) => {
      const elapsed = currentTime - state.startTime;
      const progress = Math.min(1, elapsed / state.duration);
      
      if (progress < 1) {
        hasActiveAnimations = true;
        const easedProgress = ANIMATION_CONFIG.EASING(progress);
        const currentValue = state.startValue + (state.endValue - state.startValue) * easedProgress;
        state.onUpdate(currentValue);
      } else {
        state.onUpdate(state.endValue);
        state.resolve();
        this.animations.delete(id);
      }
    });

    if (hasActiveAnimations) {
      this.frameId = requestAnimationFrame(this.update.bind(this));
    } else {
      this.frameId = null;
    }
  }

  /**
   * Cancels an ongoing animation
   * @param {string} id - Animation identifier
   */
  cancel(id) {
    if (this.animations.has(id)) {
      const animation = this.animations.get(id);
      animation.resolve();
      this.animations.delete(id);
    }
  }

  /**
   * Creates a sliding transition animation
   * @param {HTMLElement} element - Element to animate
   * @param {number} startX - Starting X position
   * @param {number} endX - Ending X position
   * @param {string} id - Animation identifier
   * @returns {Promise<void>}
   */
  slide(element, startX, endX, id) {
    return this.animate(id, {
      duration: WIDGET_CONFIG.TRANSITION_DURATION,
      startValue: startX,
      endValue: endX,
      onUpdate: (value) => {
        element.style.transform = `translateX(${value}px)`;
      }
    });
  }

  /**
   * Creates a fade transition animation
   * @param {HTMLElement} element - Element to animate
   * @param {number} startOpacity - Starting opacity
   * @param {number} endOpacity - Ending opacity
   * @param {string} id - Animation identifier
   * @returns {Promise<void>}
   */
  fade(element, startOpacity, endOpacity, id) {
    return this.animate(id, {
      duration: WIDGET_CONFIG.TRANSITION_DURATION,
      startValue: startOpacity,
      endValue: endOpacity,
      onUpdate: (value) => {
        element.style.opacity = value;
      }
    });
  }

  /**
   * Performs a smooth scroll animation
   * @param {HTMLElement} element - Element to scroll
   * @param {number} targetScroll - Target scroll position
   * @param {string} id - Animation identifier
   * @returns {Promise<void>}
   */
  smoothScroll(element, targetScroll, id) {
    return this.animate(id, {
      duration: WIDGET_CONFIG.TRANSITION_DURATION,
      startValue: element.scrollLeft,
      endValue: targetScroll,
      onUpdate: (value) => {
        element.scrollLeft = value;
      }
    });
  }
}