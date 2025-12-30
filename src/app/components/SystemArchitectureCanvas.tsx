import React, { useEffect, useRef } from 'react';
import { useArchitecture } from '../context/ArchitectureContext';

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

const SystemArchitectureCanvas: React.FC<{ view?: 'overview' | 'pipeline' }> = ({ view = 'pipeline' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { state } = useArchitecture();
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number>();

  // Define nodes with proper spacing to avoid text overlap
  const nodes: Node[] = [
    // Layer 1: Input Processing (Green) - More vertical space
    { id: 'video-input', x: 100, y: 80, label: 'Video Input', type: 'input' },
    { id: 'frame-sampler', x: 100, y: 160, label: 'Frame Sampler', type: 'input' },
    { id: 'face-detector', x: 100, y: 240, label: 'Face Detector', type: 'input' },
    { id: 'audio-extractor', x: 100, y: 320, label: 'Audio Extract', type: 'input' },

    // Layer 2: Model Bank (Blue) - Better spacing
    { id: 'bg-model', x: 320, y: 60, label: 'BG-Model', type: 'model' },
    { id: 'av-model', x: 320, y: 120, label: 'AV-Model', type: 'model' },
    { id: 'cm-model', x: 320, y: 180, label: 'CM-Model', type: 'model' },
    { id: 'rr-model', x: 320, y: 240, label: 'RR-Model', type: 'model' },
    { id: 'll-model', x: 320, y: 300, label: 'LL-Model', type: 'model' },
    { id: 'tm-model', x: 320, y: 360, label: 'TM-Model', type: 'model' },

    // Layer 3: Agentic Intelligence (Purple) - Better spacing
    { id: 'routing-engine', x: 540, y: 80, label: 'Routing', type: 'agent' },
    { id: 'langgraph', x: 540, y: 160, label: 'LangGraph', type: 'agent' },
    { id: 'aggregator', x: 540, y: 240, label: 'Aggregator', type: 'agent' },
    { id: 'explainer', x: 540, y: 320, label: 'Explainer', type: 'agent' },

    // Layer 4: Output (Orange) - Better spacing
    { id: 'api-response', x: 760, y: 160, label: 'API Response', type: 'output' },
    { id: 'heatmap', x: 760, y: 240, label: 'Heatmap', type: 'output' },
  ];

  const connections: Connection[] = [
    { from: 'video-input', to: 'frame-sampler' },
    { from: 'video-input', to: 'audio-extractor' },
    { from: 'frame-sampler', to: 'face-detector' },
    { from: 'face-detector', to: 'bg-model' },
    { from: 'face-detector', to: 'av-model' },
    { from: 'face-detector', to: 'cm-model' },
    { from: 'face-detector', to: 'rr-model' },
    { from: 'face-detector', to: 'll-model' },
    { from: 'face-detector', to: 'tm-model' },
    { from: 'audio-extractor', to: 'av-model' },
    { from: 'bg-model', to: 'routing-engine' },
    { from: 'routing-engine', to: 'langgraph' },
    { from: 'av-model', to: 'aggregator' },
    { from: 'cm-model', to: 'aggregator' },
    { from: 'rr-model', to: 'aggregator' },
    { from: 'll-model', to: 'aggregator' },
    { from: 'tm-model', to: 'aggregator' },
    { from: 'langgraph', to: 'aggregator' },
    { from: 'aggregator', to: 'explainer' },
    { from: 'explainer', to: 'api-response' },
    { from: 'explainer', to: 'heatmap' },
  ];

  const getNodeColor = (type: string, isActive: boolean) => {
    const colors = {
      input: isActive ? '#10B981' : '#059669',
      model: isActive ? '#3B82F6' : '#2563EB',
      agent: isActive ? '#8B5CF6' : '#7C3AED',
      output: isActive ? '#F59E0B' : '#D97706',
    };
    return colors[type] || '#6B7280';
  };

  // Spawn particles - exactly like original
  useEffect(() => {
    if (state.processingStage === 'idle') return;

    const interval = setInterval(() => {
      connections.forEach(conn => {
        if (state.activeModels.includes(conn.from) && Math.random() < 0.3) {
          const fromNode = nodes.find(n => n.id === conn.from);
          const toNode = nodes.find(n => n.id === conn.to);

          if (fromNode && toNode) {
            particlesRef.current.push({
              id: `${conn.from}-${conn.to}-${Date.now()}`,
              fromX: fromNode.x,
              fromY: fromNode.y,
              toX: toNode.x,
              toY: toNode.y,
              progress: 0,
              speed: 0.008 + Math.random() * 0.012,
              color: getNodeColor(fromNode.type, true),
            });
          }
        }
      });
    }, 150);

    return () => clearInterval(interval);
  }, [state.processingStage, state.activeModels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections - exactly like original
      connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);

        if (!fromNode || !toNode) return;

        const isActive = state.activeModels.includes(conn.from) || state.activeModels.includes(conn.to);

        ctx.strokeStyle = isActive ? 'rgba(59, 130, 246, 0.4)' : 'rgba(100, 116, 139, 0.2)';
        ctx.lineWidth = isActive ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();
      });

      // Update and draw particles - exactly like original
      particlesRef.current = particlesRef.current.filter(particle => {
        particle.progress += particle.speed;
        if (particle.progress > 1) return false;

        const x = particle.fromX + (particle.toX - particle.fromX) * particle.progress;
        const y = particle.fromY + (particle.toY - particle.fromY) * particle.progress;

        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 10);
        gradient.addColorStop(0, particle.color);
        gradient.addColorStop(1, 'transparent');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();

        return true;
      });

      // Draw nodes - exactly like original but with better spacing
      nodes.forEach(node => {
        const isActive = state.activeModels.includes(node.id);

        // Glow effect
        if (isActive) {
          const glowGradient = ctx.createRadialGradient(node.x, node.y, 15, node.x, node.y, 40);
          glowGradient.addColorStop(0, getNodeColor(node.type, true) + '40');
          glowGradient.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGradient;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 40, 0, Math.PI * 2);
          ctx.fill();
        }

        // Node circle
        ctx.fillStyle = isActive
          ? getNodeColor(node.type, true)
          : 'rgba(55, 65, 81, 0.8)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = isActive ? '#FFFFFF' : 'rgba(148, 163, 184, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label with proper spacing and visibility
        ctx.fillStyle = isActive ? '#FFFFFF' : 'rgba(30, 41, 59, 0.9)'; // Dark text for better visibility
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        
        // Add text background for better readability
        const textWidth = ctx.measureText(node.label).width;
        const padding = 4;
        const bgY = node.y + 45;
        
        // Background rectangle
        ctx.fillStyle = isActive ? 'rgba(0, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.9)';
        ctx.fillRect(node.x - textWidth/2 - padding, bgY - 8, textWidth + padding * 2, 16);
        
        // Text
        ctx.fillStyle = isActive ? '#FFFFFF' : 'rgba(30, 41, 59, 0.9)';
        ctx.fillText(node.label, node.x, bgY + 4);
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
  }, [state.activeModels]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
};

export default SystemArchitectureCanvas;