import { useEffect, useRef } from 'react';

export type ReidDisplayMode = 'overlay' | 'heatmap' | 'original';

interface Props {
  imageUrl: string;
  values: number[][];
  mode: ReidDisplayMode;
  opacity: number;
  label: string;
}

const SIZE = 256;
const BODY_WIDTH = 128;
const BODY_X = (SIZE - BODY_WIDTH) / 2;

const COLOR_STOPS: Array<[number, [number, number, number]]> = [
  [0, [40, 65, 180]],
  [0.25, [20, 170, 220]],
  [0.5, [70, 210, 120]],
  [0.75, [250, 215, 45]],
  [1, [220, 35, 35]],
];

function colorFor(value: number): [number, number, number] {
  const bounded = Math.max(0, Math.min(1, value));
  for (let index = 1; index < COLOR_STOPS.length; index += 1) {
    const [rightAt, right] = COLOR_STOPS[index];
    const [leftAt, left] = COLOR_STOPS[index - 1];
    if (bounded <= rightAt) {
      const ratio = (bounded - leftAt) / (rightAt - leftAt);
      return left.map((channel, channelIndex) =>
        Math.round(channel + (right[channelIndex] - channel) * ratio),
      ) as [number, number, number];
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1];
}

function drawHeatmap(
  context: CanvasRenderingContext2D,
  values: number[][],
  alpha: number,
) {
  const source = document.createElement('canvas');
  source.width = 5;
  source.height = 9;
  const sourceContext = source.getContext('2d');
  if (!sourceContext) return;
  const pixels = sourceContext.createImageData(5, 9);
  for (let row = 0; row < 9; row += 1) {
    for (let column = 0; column < 5; column += 1) {
      const [red, green, blue] = colorFor(values[row][column]);
      const offset = (row * 5 + column) * 4;
      pixels.data[offset] = red;
      pixels.data[offset + 1] = green;
      pixels.data[offset + 2] = blue;
      pixels.data[offset + 3] = 255;
    }
  }
  sourceContext.putImageData(pixels, 0, 0);
  context.save();
  context.globalAlpha = alpha;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, BODY_X, 0, BODY_WIDTH, SIZE);
  context.restore();
}

export default function ReidHeatmapCanvas({ imageUrl, values, mode, opacity, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return undefined;
    let cancelled = false;

    const paint = (image?: HTMLImageElement) => {
      if (cancelled) return;
      context.clearRect(0, 0, SIZE, SIZE);
      context.fillStyle = '#0f172a';
      context.fillRect(0, 0, SIZE, SIZE);
      if (image && mode !== 'heatmap') {
        context.drawImage(image, BODY_X, 0, BODY_WIDTH, SIZE);
      }
      if (mode !== 'original') {
        drawHeatmap(context, values, mode === 'overlay' ? opacity : 1);
      }
    };

    if (mode === 'heatmap') {
      paint();
      return () => { cancelled = true; };
    }
    const image = new Image();
    image.onload = () => paint(image);
    image.onerror = () => paint();
    image.src = imageUrl;
    return () => { cancelled = true; };
  }, [imageUrl, mode, opacity, values]);

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      aria-label={label}
      className="reid-heatmap-canvas"
    />
  );
}
