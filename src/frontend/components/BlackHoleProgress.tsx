import React, { useEffect, useRef } from 'react';

interface BlackHoleProgressProps {
  progress: number; // 0-100
  size?: number;
  growable?: boolean;
  minSize?: number;
  maxSize?: number;
}

interface BackgroundStar {
  angle: number;
  radiusNorm: number;
  size: number;
  alpha: number;
  hue: number;
  depth: number;
  twinkleOffset: number;
  twinkleSpeed: number;
  drift: number;
}

interface InfallStar {
  angle: number;
  radiusNorm: number;
  speed: number;
  size: number;
  alpha: number;
  hue: number;
  depth: number;
  stretch: number;
  twist: number;
}

interface DiskParticle {
  angle: number;
  orbitNorm: number;
  speed: number;
  size: number;
  alpha: number;
  hue: number;
  depth: number;
  tilt: number;
  twinkleOffset: number;
}

const TAU = Math.PI * 2;

const randomBetween = (min: number, max: number) => min + Math.random() * (max - min);

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const pickStarHue = () => {
  const roll = Math.random();
  if (roll < 0.68) return randomBetween(205, 220);
  if (roll < 0.9) return randomBetween(188, 202);
  return randomBetween(38, 52);
};

const createBackgroundStar = (): BackgroundStar => ({
  angle: randomBetween(0, TAU),
  radiusNorm: Math.pow(Math.random(), 0.65),
  size: randomBetween(0.7, 2.35),
  alpha: randomBetween(0.28, 0.95),
  hue: pickStarHue(),
  depth: randomBetween(0.18, 1),
  twinkleOffset: randomBetween(0, TAU),
  twinkleSpeed: randomBetween(0.8, 2.6),
  drift: randomBetween(-0.03, 0.03),
});

const createInfallStar = (spawnOuter = false): InfallStar => ({
  angle: randomBetween(0, TAU),
  radiusNorm: randomBetween(spawnOuter ? 0.92 : 0.24, 1.18),
  speed: randomBetween(0.45, 1.25),
  size: randomBetween(0.8, 2.5),
  alpha: randomBetween(0.32, 0.95),
  hue: Math.random() < 0.18 ? randomBetween(34, 48) : randomBetween(196, 214),
  depth: randomBetween(0.12, 1),
  stretch: randomBetween(0.7, 1.45),
  twist: randomBetween(0.01, 0.05),
});

const createDiskParticle = (): DiskParticle => ({
  angle: randomBetween(0, TAU),
  orbitNorm: randomBetween(0.22, 0.62),
  speed: randomBetween(0.45, 1.8),
  size: randomBetween(0.8, 2.7),
  alpha: randomBetween(0.26, 0.82),
  hue: Math.random() < 0.58 ? randomBetween(192, 206) : randomBetween(28, 44),
  depth: randomBetween(0.2, 1),
  tilt: randomBetween(0.12, 0.26),
  twinkleOffset: randomBetween(0, TAU),
});

export const BlackHoleProgress: React.FC<BlackHoleProgressProps> = ({
  progress,
  size: staticSize = 162,
  growable = false,
  minSize = 115,
  maxSize = 520,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const backgroundStarsRef = useRef<BackgroundStar[]>([]);
  const infallStarsRef = useRef<InfallStar[]>([]);
  const diskParticlesRef = useRef<DiskParticle[]>([]);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);
  const cameraRef = useRef({ x: 0, y: 0 });
  const targetProgressRef = useRef(progress / 100);
  const displayProgressRef = useRef(progress / 100);
  const currentSizeRef = useRef(growable ? minSize : staticSize);

  targetProgressRef.current = progress / 100;

  const computeSize = (p: number) => {
    if (!growable) return staticSize;
    const t = Math.min(1, p);
    const eased = 1 - Math.pow(1 - t, 2);
    return minSize + (maxSize - minSize) * eased;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const canvasW = growable ? maxSize : staticSize;
    const canvasH = growable ? maxSize : staticSize;
    const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;

    canvas.width = Math.round(canvasW * dpr);
    canvas.height = Math.round(canvasH * dpr);
    canvas.style.width = `${canvasW}px`;
    canvas.style.height = `${canvasH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    backgroundStarsRef.current = Array.from({ length: 420 }, createBackgroundStar);
    infallStarsRef.current = Array.from({ length: 260 }, () => createInfallStar(false));
    diskParticlesRef.current = Array.from({ length: 240 }, createDiskParticle);

    const animate = () => {
      const lerpSpeed = 0.08;
      displayProgressRef.current += (targetProgressRef.current - displayProgressRef.current) * lerpSpeed;
      const currentProgress = displayProgressRef.current;

      const targetSize = computeSize(currentProgress);
      currentSizeRef.current += (targetSize - currentSizeRef.current) * 0.06;
      const curSize = currentSizeRef.current;
      const circleRadius = curSize / 2;
      const fieldRadius = circleRadius * 0.975;
      const holeRadius = circleRadius * (0.17 + currentProgress * 0.07);
      const diskRadius = holeRadius * (2.15 + currentProgress * 0.55);

      ctx.clearRect(0, 0, canvasW, canvasH);

      const cx = canvasW / 2;
      const cy = circleRadius;

      timeRef.current += 0.012 + currentProgress * 0.026;

      const sway = fieldRadius * (0.018 + currentProgress * 0.03);
      const targetCamX = Math.sin(timeRef.current * 0.7) * sway;
      const targetCamY = Math.cos(timeRef.current * 0.95) * sway;
      cameraRef.current.x += (targetCamX - cameraRef.current.x) * 0.06;
      cameraRef.current.y += (targetCamY - cameraRef.current.y) * 0.06;

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, circleRadius, 0, TAU);
      ctx.clip();

      const bgGradient = ctx.createRadialGradient(cx, cy, holeRadius * 0.4, cx, cy, fieldRadius * 1.08);
      bgGradient.addColorStop(0, 'rgba(1, 3, 8, 0.92)');
      bgGradient.addColorStop(0.48, 'rgba(3, 7, 16, 0.96)');
      bgGradient.addColorStop(1, 'rgba(0, 0, 0, 1)');
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, canvasW, canvasH);

      ctx.fillStyle = `rgba(2, 6, 16, ${0.1 + currentProgress * 0.12})`;
      ctx.fillRect(0, 0, canvasW, canvasH);

      backgroundStarsRef.current.forEach((star) => {
        const twinkle = 0.58 + 0.42 * Math.sin(timeRef.current * star.twinkleSpeed + star.twinkleOffset);
        const radialDrift = Math.sin(timeRef.current * 0.55 + star.twinkleOffset) * fieldRadius * 0.014;
        const radial = clamp(star.radiusNorm * fieldRadius + radialDrift, holeRadius * 1.75, fieldRadius * 0.995);
        const driftAngle = star.angle + timeRef.current * star.drift;
        const parallax = 1 + (1 - star.depth) * 0.16;
        const x = cx + Math.cos(driftAngle) * radial * parallax - cameraRef.current.x * star.depth * 0.18;
        const y = cy + Math.sin(driftAngle) * radial * (0.94 + star.depth * 0.05) * parallax - cameraRef.current.y * star.depth * 0.18;
        const alpha = star.alpha * (0.34 + twinkle * 0.82);
        const haloSize = star.size * (1.6 + twinkle * 0.9);

        ctx.beginPath();
        ctx.arc(x, y, haloSize, 0, TAU);
        ctx.fillStyle = `hsla(${star.hue}, 100%, 78%, ${alpha * 0.14})`;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, star.size, 0, TAU);
        ctx.fillStyle = `hsla(${star.hue}, 100%, 92%, ${alpha})`;
        ctx.fill();
      });

      ctx.save();
      ctx.globalCompositeOperation = 'screen';

      diskParticlesRef.current.forEach((particle) => {
        const orbit = clamp(fieldRadius * particle.orbitNorm, holeRadius * 1.15, fieldRadius * 0.72);
        const spin = timeRef.current * particle.speed * (0.75 + currentProgress * 1.3);
        const angle = particle.angle + spin;
        const flatten = particle.tilt + currentProgress * 0.055;
        const x = cx + Math.cos(angle) * orbit;
        const y = cy + Math.sin(angle) * orbit * flatten;
        const tailAngle = angle - 0.22 * particle.speed;
        const tailX = cx + Math.cos(tailAngle) * orbit * 1.04;
        const tailY = cy + Math.sin(tailAngle) * orbit * flatten;
        const sparkle = 0.55 + 0.45 * Math.sin(timeRef.current * 1.8 + particle.twinkleOffset);
        const alpha = particle.alpha * (0.32 + sparkle * 0.9);

        const trail = ctx.createLinearGradient(tailX, tailY, x, y);
        trail.addColorStop(0, `hsla(${particle.hue}, 100%, 60%, 0)`);
        trail.addColorStop(0.45, `hsla(${particle.hue}, 100%, 68%, ${alpha * 0.18})`);
        trail.addColorStop(1, `hsla(${particle.hue}, 100%, 76%, ${alpha})`);

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = trail;
        ctx.lineWidth = particle.size * (0.7 + (1 - particle.depth) * 0.7);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x, y, particle.size * (0.8 + sparkle * 0.25), 0, TAU);
        ctx.fillStyle = `hsla(${particle.hue}, 100%, 78%, ${alpha * 0.7})`;
        ctx.fill();
      });

      infallStarsRef.current.forEach((star) => {
        star.radiusNorm -= star.speed * (0.0035 + currentProgress * 0.011) * (1.1 - star.depth * 0.35);
        star.angle += star.twist * (0.85 + currentProgress * 2.2);

        if (star.radiusNorm * fieldRadius <= holeRadius * 0.9) {
          Object.assign(star, createInfallStar(true));
        }

        const radial = star.radiusNorm * fieldRadius;
        const curvature = clamp((diskRadius - radial) / diskRadius, 0, 1);
        const warpedAngle = star.angle + curvature * 0.95;
        const ellipse = 0.9 - (1 - star.depth) * 0.08;
        const headX = cx + Math.cos(warpedAngle) * radial - cameraRef.current.x * 0.08;
        const headY = cy + Math.sin(warpedAngle) * radial * ellipse - cameraRef.current.y * 0.08;
        const trailRadiusNorm = Math.min(1.22, star.radiusNorm + star.stretch * (0.016 + currentProgress * 0.023));
        const tailAngle = warpedAngle - star.twist * 18;
        const tailRadius = trailRadiusNorm * fieldRadius;
        const tailX = cx + Math.cos(tailAngle) * tailRadius;
        const tailY = cy + Math.sin(tailAngle) * tailRadius * ellipse;
        const alpha = star.alpha * (0.42 + curvature * 0.9);

        const trail = ctx.createLinearGradient(tailX, tailY, headX, headY);
        trail.addColorStop(0, `hsla(${star.hue}, 100%, 65%, 0)`);
        trail.addColorStop(0.38, `hsla(${star.hue}, 100%, 70%, ${alpha * 0.16})`);
        trail.addColorStop(1, `hsla(${star.hue}, 100%, 85%, ${alpha})`);

        ctx.beginPath();
        ctx.moveTo(tailX, tailY);
        ctx.lineTo(headX, headY);
        ctx.strokeStyle = trail;
        ctx.lineWidth = star.size * (0.85 + curvature * 1.15);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(headX, headY, star.size * (0.7 + curvature * 0.8), 0, TAU);
        ctx.fillStyle = `hsla(${star.hue}, 100%, 92%, ${Math.min(1, alpha + 0.05)})`;
        ctx.fill();
      });

      const ringGlow = ctx.createRadialGradient(cx, cy, holeRadius * 0.72, cx, cy, holeRadius * 4.2);
      ringGlow.addColorStop(0, 'rgba(0, 0, 0, 0)');
      ringGlow.addColorStop(0.22, `rgba(251, 191, 36, ${0.18 + currentProgress * 0.12})`);
      ringGlow.addColorStop(0.42, `rgba(56, 189, 248, ${0.16 + currentProgress * 0.12})`);
      ringGlow.addColorStop(1, 'rgba(56, 189, 248, 0)');
      ctx.fillStyle = ringGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, holeRadius * 4.2, 0, TAU);
      ctx.fill();

      ctx.restore();

      ctx.beginPath();
      ctx.arc(cx, cy, holeRadius * 1.08, 0, TAU);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
      ctx.fill();

      const photonRing = ctx.createRadialGradient(cx, cy, holeRadius * 0.82, cx, cy, holeRadius * 1.45);
      photonRing.addColorStop(0, 'rgba(0, 0, 0, 0)');
      photonRing.addColorStop(0.2, `rgba(255, 208, 120, ${0.28 + currentProgress * 0.12})`);
      photonRing.addColorStop(0.48, `rgba(125, 211, 252, ${0.22 + currentProgress * 0.16})`);
      photonRing.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = photonRing;
      ctx.beginPath();
      ctx.arc(cx, cy, holeRadius * 1.5, 0, TAU);
      ctx.fill();

      const coreGradient = ctx.createRadialGradient(cx, cy, holeRadius * 0.08, cx, cy, holeRadius);
      coreGradient.addColorStop(0, 'rgba(0, 0, 0, 0.98)');
      coreGradient.addColorStop(0.7, 'rgba(0, 0, 0, 1)');
      coreGradient.addColorStop(1, 'rgba(3, 6, 12, 1)');
      ctx.beginPath();
      ctx.arc(cx, cy, holeRadius, 0, TAU);
      ctx.fillStyle = coreGradient;
      ctx.fill();

      const vignette = ctx.createRadialGradient(cx, cy, holeRadius * 1.1, cx, cy, fieldRadius * 1.05);
      vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
      vignette.addColorStop(0.72, 'rgba(0, 0, 0, 0.05)');
      vignette.addColorStop(1, 'rgba(0, 0, 0, 0.42)');
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, canvasW, canvasH);

      ctx.restore();

      const margin = 4;
      const ringR = circleRadius - margin;
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + TAU * currentProgress;
      const light = 56 + currentProgress * 14;
      const color = `hsl(197, 92%, ${light}%)`;

      ctx.beginPath();
      ctx.arc(cx, cy, ringR, startAngle, endAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.lineCap = 'butt';
      ctx.shadowBlur = 14 + currentProgress * 8;
      ctx.shadowColor = color;
      ctx.stroke();
      ctx.shadowBlur = 0;

      const percent = Math.round(currentProgress * 100);
      const baseFontSize = (growable ? minSize : staticSize) * 0.22;
      const growFactor = growable ? Math.min(1.4, curSize / minSize * 0.6 + 0.4) : 1;
      const fontSize = baseFontSize * growFactor;
      const textY = growable
        ? Math.min(cy, minSize / 2 + (curSize - minSize) * 0.08)
        : cy;

      ctx.font = `bold ${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.shadowBlur = 18;
      ctx.shadowColor = 'hsla(197, 100%, 72%, 0.8)';
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
  }, [growable, maxSize, minSize, staticSize]);

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
            width: canvasW,
            height: canvasH,
          }}
        />
      </div>
    </div>
  );
};
