/**
 * Centralized visibility rules for comparison/model-selection views.
 *
 * The comparison source files still contain these baselines for reproducibility,
 * but they are intentionally not exposed as selectable or displayed models.
 */

const HIDDEN_MODEL_NAMES = new Set(['GCN', 'KMEANS', 'MLP']);

export function isHiddenModelName(modelName: string): boolean {
  const normalizedName = modelName.toUpperCase().replace(/[\s_-]/g, '');
  return HIDDEN_MODEL_NAMES.has(normalizedName);
}
