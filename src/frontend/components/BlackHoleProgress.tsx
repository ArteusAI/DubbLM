import React, { useRef, useEffect } from 'react';

interface BlackHoleProgressProps {
  progress: number; // 0-100
  size?: number;
  // Dynamic growth settings
  growable?: boolean;
  minSize?: number;
  maxSize?: number;
}

interface Star {
  x: number;
  y: number;
  z: number;
  pz: number;
}

export const BlackHoleProgress: React.FC<BlackHoleProgressProps> = ({
  progress,
  size: staticSize = 162,
  growable = false,
  minSize = 115,
  maxSize = 520,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const starsRef = useRef<Star[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);
  const cameraRef = useRef({ x: 0, y: 0 });
  const targetProgressRef = useRef(progress / 100);
  const displayProgressRef = useRef(progress / 100);
  const currentSizeRef = useRef(growable ? minSize : staticSize);

  targetProgressRef.current = progress / 100;

  // Compute current size based on progress
  const computeSize = (p: number) => {
    if (!growable) return staticSize;
    const t = Math.min(1, p);
    const eased = 1 - Math.pow(1 - t, 2);
    return minSize + (maxSize - minSize) * eased;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Canvas is always maxSize width, maxSize height for growable
    const canvasW = growable ? maxSize : staticSize;
    const canvasH = growable ? maxSize : staticSize;
    canvas.width = canvasW;
    canvas.height = canvasH;

    const numStars = 1800;
    const maxDepth = canvasW * 4;
    const starSpread = canvasW * 8;

    starsRef.current = [];
    for (let i = 0; i < numStars; i++) {
      const star: Star = {
        x: (Math.random() - 0.5) * starSpread,
        y: (Math.random() - 0.5) * starSpread,
        z: Math.random() * maxDepth,
        pz: 0,
      };
      star.pz = star.z;
      starsRef.current.push(star);
    }

    const animate = () => {
      const lerpSpeed = 0.08;
      displayProgressRef.current += (targetProgressRef.current - displayProgressRef.current) * lerpSpeed;
      const currentProgress = displayProgressRef.current;

      // Compute dynamic size
      const targetSize = computeSize(currentProgress);
      currentSizeRef.current += (targetSize - currentSizeRef.current) * 0.06;
      const curSize = currentSizeRef.current;
      const circleRadius = curSize / 2;

      ctx.clearRect(0, 0, canvasW, canvasH);

      // Circle center: horizontally centered, vertically anchored so TOP edge stays at y=0
      // center = (canvasW/2, circleRadius) => top of circle is at y=0
      const cx = canvasW / 2;
      const cy = circleRadius;

      // Clip to circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, circleRadius, 0, Math.PI * 2);
      ctx.clip();

      // Background
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvasW, canvasH);

      // Fade effect
      ctx.fillStyle = `rgba(0, 0, 0, ${0.45 - (currentProgress * 0.2)})`;
      ctx.fillRect(0, 0, canvasW, canvasH);

      timeRef.current += 0.01 + (currentProgress * 0.04);

      // Camera movement
      const turnIntensity = 30 + (currentProgress * 40);
      const targetCamX = Math.sin(timeRef.current) * turnIntensity;
      const targetCamY = Math.cos(timeRef.current * 1.3) * turnIntensity;
      cameraRef.current.x += (targetCamX - cameraRef.current.x) * 0.08;
      cameraRef.current.y += (targetCamY - cameraRef.current.y) * 0.08;

      // Speed based on progress
      const baseSpeed = 0.8;
      const warpSpeed = baseSpeed + (currentProgress * 6);
      const rotation = timeRef.current * 0.2 * currentProgress;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const fov = 130;

      // Draw stars
      starsRef.current.forEach(star => {
        star.z -= warpSpeed;

        let rx = star.x * cos - star.y * sin;
        let ry = star.x * sin + star.y * cos;

        if (star.z <= 1) {
          star.z = maxDepth;
          star.pz = star.z;
          star.x = (Math.random() - 0.5) * starSpread;
          star.y = (Math.random() - 0.5) * starSpread;
          rx = star.x;
          ry = star.y;
        }

        const scale = fov / star.z;
        const x2d = cx + (rx - cameraRef.current.x) * scale;
        const y2d = cy + (ry - cameraRef.current.y) * scale;
        const scaleP = fov / (star.z + warpSpeed * 0.8);
        const x2d_prev = cx + (rx - cameraRef.current.x) * scaleP;
        const y2d_prev = cy + (ry - cameraRef.current.y) * scaleP;

        const distRatio = star.z / maxDepth;
        const alpha = Math.min(1, Math.max(0.15, (1 - distRatio) * 5));

        let r = 255, g = 255, b = 255;
        if (currentProgress > 0.3) { r = 56; g = 189; b = 248; }
        if (currentProgress > 0.7) { r = 14; g = 165; b = 233; }

        ctx.beginPath();
        ctx.moveTo(x2d_prev, y2d_prev);
        ctx.lineTo(x2d, y2d);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.lineWidth = Math.min(2.5, scale * 0.6);
        ctx.stroke();
      });

      ctx.restore(); // Remove clip

      // Draw progress ring
      const margin = 4;
      const ringR = circleRadius - margin;
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + (Math.PI * 2 * currentProgress);
      const light = 50 + (currentProgress * 15);
      const color = `hsl(199, 90%, ${light}%)`;

      ctx.beginPath();
      ctx.arc(cx, cy, ringR, startAngle, endAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.lineCap = 'butt';
      ctx.shadowBlur = 12 + (currentProgress * 5);
      ctx.shadowColor = color;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw percentage text
      // As circle grows, the percentage stays near the top visible area
      // It shifts down slightly but never past the original circle size position
      const percent = Math.round(currentProgress * 100);
      const baseFontSize = (growable ? minSize : staticSize) * 0.22;
      const growFactor = growable ? Math.min(1.4, curSize / minSize * 0.6 + 0.4) : 1;
      const fontSize = baseFontSize * growFactor;
      // Text Y: starts at circle center for small circle, but as it grows
      // the text stays near the top portion (around minSize/2 from top)
      const textY = growable
        ? Math.min(cy, minSize / 2 + (curSize - minSize) * 0.08)
        : cy;

      ctx.font = `bold ${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.shadowBlur = 15;
      ctx.shadowColor = `hsl(199, 90%, 60%)`;
      ctx.fillStyle = '#fff';
      ctx.fillText(`${percent}%`, cx, textY);
      ctx.shadowBlur = 0;

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [growable ? maxSize : staticSize]);

  // Shake effect for high progress (>80%)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    if (progress > 80) {
      const intensity = (progress - 80) * 0.6;
      const tx = (Math.random() - 0.5) * intensity;
      const ty = (Math.random() - 0.5) * intensity;
      container.style.transform = `translate(${tx}px, ${ty}px)`;
    } else {
      container.style.transform = 'none';
    }
  }, [progress]);

  const canvasW = growable ? maxSize : staticSize;
  const canvasH = growable ? maxSize : staticSize;

  return (
    <div
      style={{
        animation: growable ? 'none' : 'pulse-subtle 3s ease-in-out infinite',
      }}
    >
      <style>{`
        @keyframes pulse-subtle {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.03); }
        }
      `}</style>
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          width: canvasW,
          height: canvasH,
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
          }}
        />
      </div>
    </div>
  );
};
