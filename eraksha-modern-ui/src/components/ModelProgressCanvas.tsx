import React, { useEffect, useRef } from 'react';
import { useArchitecture } from '../context/ArchitectureContext';

// Extend CanvasRenderingContext2D to include roundRect
declare global {
  interface CanvasRenderingContext2D {
    roundRect?: (x: number, y: number, width: number, height: number, radius: number) => void;
  }
}

interface Node {
  id: string;
  x: number;
  y: number;
  label: string;
  type: 'input' | 'model' | 'agent' | 'output';
}

interface Connection {
  from: string;
  to: string;
}

interface Particle {
  id: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
  color: string;
}

const ModelProgressCanvas: React.FC<{ view?: 'overview' | 'pipeline' }> = ({ view = 'pipeline' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { state } = useArchitecture();
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();

  // Enhanced node layout with better spacing for full-width display
  const nodes: Node[] = [
    // Layer 1: Input Processing (Left side)
    { id: 'video-input', x: 120, y: 120, label: 'Video Input', type: 'input' },
    { id: 'video-decoder', x: 120, y: 180, label: 'Video Decoder', type: 'input' },
    { id: 'frame-sampler', x: 120, y: 240, label: 'Frame Sampler', type: 'input' },
    { id: 'face-detector', x: 120, y: 300, label: 'Face Detector', type: 'input' },
    { id: 'audio-extractor', x: 120, y: 360, label: 'Audio Extractor', type: 'input' },

    // Layer 2: Model Bank (Center-left)
    { id: 'bg-model', x: 350, y: 100, label: 'BG-Model', type: 'model' },
    { id: 'av-model', x: 350, y: 160, label: 'AV-Model', type: 'model' },
    { id: 'cm-model', x: 350, y: 220, label: 'CM-Model', type: 'model' },
    { id: 'rr-model', x: 350, y: 280, label: 'RR-Model', type: 'model' },
    { id: 'll-model', x: 350, y: 340, label: 'LL-Model', type: 'model' },
    { id: 'tm-model', x: 350, y: 400, label: 'TM-Model', type: 'model' },

    // Layer 3: Agentic Intelligence (Center-right)
    { id: 'routing-engine', x: 580, y: 120, label: 'Routing Engine', type: 'agent' },
    { id: 'langgraph', x: 580, y: 200, label: 'LangGraph Agent', type: 'agent' },
    { id: 'aggregator', x: 580, y: 280, label: 'Aggregator', type: 'agent' },
    { id: 'bias-correction', x: 580, y: 340, label: 'Bias Correction', type: 'agent' },
    { id: 'explainer', x: 580, y: 400, label: 'Explainer', type: 'agent' },

    // Layer 4: Output (Right side)
    { id: 'fastapi', x: 810, y: 200, label: 'FastAPI', type: 'output' },
    { id: 'react-ui', x: 810, y: 280, label: 'React UI', type: 'output' },
    { id: 'api-response', x: 810, y: 360, label: 'API Response', type: 'output' },
  ];

  const connections: Connection[] = [
    // Input flow
    { from: 'video-input', to: 'video-decoder' },
    { from: 'video-decoder', to: 'frame-sampler' },
    { from: 'video-decoder', to: 'audio-extractor' },
    { from: 'frame-sampler', to: 'face-detector' },
    
    // To baseline model
    { from: 'face-detector', to: 'bg-model' },
    { from: 'audio-extractor', to: 'av-model' },
    
    // Baseline to routing
    { from: 'bg-model', to: 'routing-engine' },
    { from: 'routing-engine', to: 'langgraph' },
    
    // Specialist models (conditional)
    { from: 'face-detector', to: 'av-model' },
    { from: 'face-detector', to: 'cm-model' },
    { from: 'face-detector', to: 'rr-model' },
    { from: 'face-detector', to: 'll-model' },
    { from: 'face-detector', to: 'tm-model' },
    
    // To aggregation
    { from: 'av-model', to: 'aggregator' },
    { from: 'cm-model', to: 'aggregator' },
    { from: 'rr-model', to: 'aggregator' },
    { from: 'll-model', to: 'aggregator' },
    { from: 'tm-model', to: 'aggregator' },
    { from: 'langgraph', to: 'aggregator' },
    
    // Final processing
    { from: 'aggregator', to: 'bias-correction' },
    { from: 'bias-correction', to: 'explainer' },
    { from: 'explainer', to: 'fastapi' },
    { from: 'fastapi', to: 'react-ui' },
    { from: 'react-ui', to: 'api-response' },
  ];

  const getNodeColor = (type: string, isActive: boolean) => {
    const colors = {
      input: isActive ? '#10B981' : '#059669',    // Emerald
      model: isActive ? '#3B82F6' : '#2563EB',    // Blue  
      agent: isActive ? '#8B5CF6' : '#7C3AED',    // Violet
      output: isActive ? '#F59E0B' : '#D97706',   // Amber
    };
    return colors[type as keyof typeof colors] || '#6B7280';
  };

  const getParticleColor = (type: string) => {
    const colors = {
      input: '#34D399',   // Bright emerald
      model: '#60A5FA',   // Bright blue
      agent: '#A78BFA',   // Bright violet  
      output: '#FBBF24',  // Bright amber
    };
    return colors[type as keyof typeof colors] || '#9CA3AF';
  };

  const getNodeGradient = (ctx: CanvasRenderingContext2D, x: number, y: number, type: string, isActive: boolean) => {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
    const baseColor = getNodeColor(type, isActive);
    
    if (isActive) {
      gradient.addColorStop(0, baseColor);
      gradient.addColorStop(0.7, baseColor + 'CC');
      gradient.addColorStop(1, baseColor + '44');
    } else {
      gradient.addColorStop(0, baseColor + '88');
      gradient.addColorStop(1, baseColor + '22');
    }
    
    return gradient;
  };

  // Enhanced particle spawning with better timing and colors
  useEffect(() => {
    if (state.processingStage === 'idle') return;

    const interval = setInterval(() => {
      connections.forEach(conn => {
        const shouldSpawn = state.activeModels.includes(conn.from) && 
                           (state.activeModels.includes(conn.to) || Math.random() < 0.4);
        
        if (shouldSpawn && Math.random() < 0.5) {
          const fromNode = nodes.find(n => n.id === conn.from);
          const toNode = nodes.find(n => n.id === conn.to);

          if (fromNode && toNode) {
            particlesRef.current.push({
              id: `${conn.from}-${conn.to}-${Date.now()}-${Math.random()}`,
              fromX: fromNode.x,
              fromY: fromNode.y,
              toX: toNode.x,
              toY: toNode.y,
              progress: 0,
              speed: 0.006 + Math.random() * 0.008,
              color: getParticleColor(fromNode.type),
            });
          }
        }
      });
    }, 120);

    return () => clearInterval(interval);
  }, [state.processingStage, state.activeModels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Add roundRect polyfill if not available
    if (!ctx.roundRect) {
      ctx.roundRect = function(x: number, y: number, width: number, height: number, radius: number) {
        this.beginPath();
        this.moveTo(x + radius, y);
        this.lineTo(x + width - radius, y);
        this.quadraticCurveTo(x + width, y, x + width, y + radius);
        this.lineTo(x + width, y + height - radius);
        this.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        this.lineTo(x + radius, y + height);
        this.quadraticCurveTo(x, y + height, x, y + height - radius);
        this.lineTo(x, y + radius);
        this.quadraticCurveTo(x, y, x + radius, y);
        this.closePath();
      };
    }

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw background grid (subtle)
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
      ctx.lineWidth = 0.5;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Draw connections with enhanced styling
      connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);

        if (!fromNode || !toNode) return;

        const isActive = state.activeModels.includes(conn.from) && state.activeModels.includes(conn.to);
        const isPartialActive = state.activeModels.includes(conn.from) || state.activeModels.includes(conn.to);

        // Connection line
        ctx.strokeStyle = isActive 
          ? 'rgba(59, 130, 246, 0.6)' 
          : isPartialActive 
            ? 'rgba(59, 130, 246, 0.3)'
            : 'rgba(100, 116, 139, 0.15)';
        ctx.lineWidth = isActive ? 3 : isPartialActive ? 2 : 1;
        
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();

        // Arrow head for active connections
        if (isActive || isPartialActive) {
          const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x);
          const arrowLength = 12;
          const arrowAngle = Math.PI / 6;

          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.moveTo(
            toNode.x - arrowLength * Math.cos(angle - arrowAngle),
            toNode.y - arrowLength * Math.sin(angle - arrowAngle)
          );
          ctx.lineTo(toNode.x, toNode.y);
          ctx.lineTo(
            toNode.x - arrowLength * Math.cos(angle + arrowAngle),
            toNode.y - arrowLength * Math.sin(angle + arrowAngle)
          );
          ctx.closePath();
          ctx.fill();
        }
      });

      // Update and draw particles with enhanced effects
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.progress += particle.speed;
        if (particle.progress > 1) return false;

        const x = particle.fromX + (particle.toX - particle.fromX) * particle.progress;
        const y = particle.fromY + (particle.toY - particle.fromY) * particle.progress;

        // Enhanced particle with multiple glow layers
        const outerGradient = ctx.createRadialGradient(x, y, 0, x, y, 20);
        outerGradient.addColorStop(0, particle.color + '40');
        outerGradient.addColorStop(0.3, particle.color + '20');
        outerGradient.addColorStop(1, 'transparent');

        ctx.fillStyle = outerGradient;
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, Math.PI * 2);
        ctx.fill();

        // Middle glow
        const middleGradient = ctx.createRadialGradient(x, y, 0, x, y, 12);
        middleGradient.addColorStop(0, particle.color + 'AA');
        middleGradient.addColorStop(0.7, particle.color + '60');
        middleGradient.addColorStop(1, 'transparent');

        ctx.fillStyle = middleGradient;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();

        // Core particle with pulsing effect
        const pulseSize = 4 + Math.sin(Date.now() * 0.01 + particle.progress * 10) * 1;
        ctx.fillStyle = particle.color;
        ctx.shadowColor = particle.color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(x, y, pulseSize, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Trailing effect
        const trailLength = 3;
        for (let i = 1; i <= trailLength; i++) {
          const trailProgress = Math.max(0, particle.progress - i * 0.02);
          if (trailProgress > 0) {
            const trailX = particle.fromX + (particle.toX - particle.fromX) * trailProgress;
            const trailY = particle.fromY + (particle.toY - particle.fromY) * trailProgress;
            const alpha = (1 - i / trailLength) * 0.6;
            
            ctx.fillStyle = particle.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            ctx.beginPath();
            ctx.arc(trailX, trailY, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        return true;
      });

      // Draw nodes with enhanced styling
      nodes.forEach(node => {
        const isActive = state.activeModels.includes(node.id);

        // Enhanced glow effect for active nodes
        if (isActive) {
          const time = Date.now() * 0.003;
          
          // Outer glow
          const outerGlow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 60);
          outerGlow.addColorStop(0, getNodeColor(node.type, true) + '30');
          outerGlow.addColorStop(0.5, getNodeColor(node.type, true) + '15');
          outerGlow.addColorStop(1, 'transparent');
          ctx.fillStyle = outerGlow;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 60, 0, Math.PI * 2);
          ctx.fill();

          // Pulsing rings
          for (let i = 0; i < 2; i++) {
            const pulseRadius = 25 + Math.sin(time + i * Math.PI) * 8;
            const alpha = (Math.sin(time + i * Math.PI) + 1) * 0.3;
            ctx.strokeStyle = getNodeColor(node.type, true) + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(node.x, node.y, pulseRadius, 0, Math.PI * 2);
            ctx.stroke();
          }
        }

        // Node circle with enhanced gradient
        const nodeGradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, 22);
        if (isActive) {
          nodeGradient.addColorStop(0, getNodeColor(node.type, true));
          nodeGradient.addColorStop(0.7, getNodeColor(node.type, true) + 'DD');
          nodeGradient.addColorStop(1, getNodeColor(node.type, true) + '88');
        } else {
          nodeGradient.addColorStop(0, getNodeColor(node.type, false) + '88');
          nodeGradient.addColorStop(1, getNodeColor(node.type, false) + '44');
        }
        
        ctx.fillStyle = nodeGradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 22, 0, Math.PI * 2);
        ctx.fill();

        // Node border with glow
        if (isActive) {
          ctx.shadowColor = '#FFFFFF';
          ctx.shadowBlur = 8;
        }
        ctx.strokeStyle = isActive ? '#FFFFFF' : 'rgba(148, 163, 184, 0.6)';
        ctx.lineWidth = isActive ? 3 : 2;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 22, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Enhanced node icon
        ctx.fillStyle = isActive ? '#FFFFFF' : 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        switch (node.type) {
          case 'input':
            // Square with rounded corners
            const size = 8;
            if (ctx.roundRect) {
              ctx.roundRect(node.x - size, node.y - size, size * 2, size * 2, 2);
            } else {
              ctx.rect(node.x - size, node.y - size, size * 2, size * 2);
            }
            break;
          case 'model':
            // Circle
            ctx.arc(node.x, node.y, 7, 0, Math.PI * 2);
            break;
          case 'agent':
            // Triangle
            ctx.moveTo(node.x, node.y - 9);
            ctx.lineTo(node.x + 8, node.y + 5);
            ctx.lineTo(node.x - 8, node.y + 5);
            ctx.closePath();
            break;
          case 'output':
            // Diamond
            ctx.moveTo(node.x, node.y - 8);
            ctx.lineTo(node.x + 8, node.y);
            ctx.lineTo(node.x, node.y + 8);
            ctx.lineTo(node.x - 8, node.y);
            ctx.closePath();
            break;
        }
        ctx.fill();

        // Enhanced label with better background
        const labelY = node.y + 40;
        ctx.font = 'bold 12px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        
        // Label background with rounded corners
        const textWidth = ctx.measureText(node.label).width;
        const padding = 6;
        const bgHeight = 18;
        
        ctx.fillStyle = isActive 
          ? 'rgba(0, 0, 0, 0.9)' 
          : 'rgba(0, 0, 0, 0.7)';
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(node.x - textWidth/2 - padding, labelY - bgHeight/2 - 2, textWidth + padding * 2, bgHeight, 4);
        } else {
          ctx.rect(node.x - textWidth/2 - padding, labelY - bgHeight/2 - 2, textWidth + padding * 2, bgHeight);
        }
        ctx.fill();
        
        // Label text with shadow
        if (isActive) {
          ctx.shadowColor = getNodeColor(node.type, true);
          ctx.shadowBlur = 4;
        }
        ctx.fillStyle = isActive ? '#FFFFFF' : 'rgba(255, 255, 255, 0.9)';
        ctx.fillText(node.label, node.x, labelY + 2);
        ctx.shadowBlur = 0;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [state.activeModels, state.processingStage]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full rounded-lg bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800" 
    />
  );
};

export default ModelProgressCanvas;