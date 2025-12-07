import React, { useRef, useEffect } from 'react';

interface BlackHoleProgressProps {
  progress: number; // 0-100
  size?: number;
}

interface Star {
  x: number;
  y: number;
  z: number;
  pz: number;
}

export const BlackHoleProgress: React.FC<BlackHoleProgressProps> = ({ 
  progress, 
  size = 162 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const starsRef = useRef<Star[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);
  const cameraRef = useRef({ x: 0, y: 0 });
  const targetProgressRef = useRef(progress / 100);
  const displayProgressRef = useRef(progress / 100);

  targetProgressRef.current = progress / 100;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = size;
    canvas.height = size;

    const cx = size / 2;
    const cy = size / 2;
    const numStars = 800;
    const maxDepth = size * 4;
    const starSpread = size * 8; // Reduced spread for denser visible stars

    // Always reinitialize stars on mount for consistent density
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
      // Smooth interpolation towards target progress
      const lerpSpeed = 0.08;
      displayProgressRef.current += (targetProgressRef.current - displayProgressRef.current) * lerpSpeed;
      const currentProgress = displayProgressRef.current;
      
      // Clear with fade effect - more transparent at higher progress
      ctx.fillStyle = `rgba(0, 0, 0, ${0.45 - (currentProgress * 0.2)})`;
      ctx.fillRect(0, 0, size, size);

      timeRef.current += 0.01 + (currentProgress * 0.04);

      // Camera movement
      const turnIntensity = 30 + (currentProgress * 40);
      const targetCamX = Math.sin(timeRef.current) * turnIntensity;
      const targetCamY = Math.cos(timeRef.current * 1.3) * turnIntensity;
      cameraRef.current.x += (targetCamX - cameraRef.current.x) * 0.08;
      cameraRef.current.y += (targetCamY - cameraRef.current.y) * 0.08;

      // Speed based on progress
      const baseSpeed = 2;
      const warpSpeed = baseSpeed + (currentProgress * 40);
      const rotation = timeRef.current * 0.2 * currentProgress;
      const cos = Math.cos(rotation);
      const sin = Math.sin(rotation);
      const fov = 130;

      // Draw stars
      starsRef.current.forEach(star => {
        star.z -= warpSpeed;

        // Apply rotation
        let rx = star.x * cos - star.y * sin;
        let ry = star.x * sin + star.y * cos;

        // Respawn star if too close
        if (star.z <= 1) {
          star.z = maxDepth;
          star.pz = star.z;
          star.x = (Math.random() - 0.5) * starSpread;
          star.y = (Math.random() - 0.5) * starSpread;
          rx = star.x;
          ry = star.y;
        }

        // Projection
        const scale = fov / star.z;
        const x2d = cx + (rx - cameraRef.current.x) * scale;
        const y2d = cy + (ry - cameraRef.current.y) * scale;
        const scaleP = fov / (star.z + warpSpeed * 0.8);
        const x2d_prev = cx + (rx - cameraRef.current.x) * scaleP;
        const y2d_prev = cy + (ry - cameraRef.current.y) * scaleP;

        // Only draw visible stars
        if (x2d > -20 && x2d < size + 20 && y2d > -20 && y2d < size + 20) {
          const distRatio = star.z / maxDepth;
          const alpha = Math.max(0, (1 - distRatio) * 1.5);

          // Color based on progress - sky blue theme
          let r = 255, g = 255, b = 255;
          if (currentProgress > 0.3) { r = 56; g = 189; b = 248; }  // brand-400 #38bdf8
          if (currentProgress > 0.7) { r = 14; g = 165; b = 233; }  // brand-500 #0ea5e9

          ctx.beginPath();
          ctx.moveTo(x2d_prev, y2d_prev);
          ctx.lineTo(x2d, y2d);
          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = Math.min(2.5, scale * 0.6);
          ctx.stroke();
        }
      });

      // Draw progress ring
      drawProgressFrame(ctx, cx, cy, size, currentProgress);
      
      // Draw percentage in center
      drawPercentage(ctx, cx, cy, currentProgress);

      animationRef.current = requestAnimationFrame(animate);
    };

    const drawProgressFrame = (
      ctx: CanvasRenderingContext2D, 
      cx: number, 
      cy: number, 
      size: number, 
      currentProgress: number
    ) => {
      const margin = 4;
      const r = (size / 2) - margin;
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + (Math.PI * 2 * currentProgress);
      
      // Sky blue brand color (hsl 199)
      const light = 50 + (currentProgress * 15);
      const color = `hsl(199, 90%, ${light}%)`;

      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, endAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.lineCap = 'butt';
      ctx.shadowBlur = 12 + (currentProgress * 5);
      ctx.shadowColor = color;
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    const drawPercentage = (
      ctx: CanvasRenderingContext2D,
      cx: number,
      cy: number,
      currentProgress: number
    ) => {
      const percent = Math.round(currentProgress * 100);
      const fontSize = size * 0.22;
      
      ctx.font = `bold ${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Glow effect - sky blue brand color
      ctx.shadowBlur = 15;
      ctx.shadowColor = `hsl(199, 90%, 60%)`;
      ctx.fillStyle = '#fff';
      ctx.fillText(`${percent}%`, cx, cy);
      ctx.shadowBlur = 0;
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [size]);

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

  return (
    <div
      className="animate-pulse-subtle"
      style={{
        animation: 'pulse-subtle 3s ease-in-out infinite',
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
          width: size,
          height: size,
          borderRadius: '50%',
          background: '#000',
          boxShadow: '0 0 35px rgba(14, 165, 233, 0.25)',
          overflow: 'hidden',
        }}
      >
        <canvas 
          ref={canvasRef} 
          style={{ 
            display: 'block', 
            borderRadius: '50%' 
          }} 
        />
      </div>
    </div>
  );
};
