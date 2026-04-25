import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const NetworkMap = ({ hovered }) => {
  const pointsRef = useRef();
  const linesRef = useRef();
  
  const particleCount = 400; 
  const maxDistance = 1.8; 
  
  // Generate random points in a large volume
  const { positions, velocities } = useMemo(() => {
    const pos = new Float32Array(particleCount * 3);
    const vel = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      // Wide 12x12x12 volume
      pos[i*3] = (Math.random() - 0.5) * 14;
      pos[i*3+1] = (Math.random() - 0.5) * 14;
      pos[i*3+2] = (Math.random() - 0.5) * 14;
      
      // Drift velocities
      vel[i*3] = (Math.random() - 0.5) * 0.015;
      vel[i*3+1] = (Math.random() - 0.5) * 0.015;
      vel[i*3+2] = (Math.random() - 0.5) * 0.015;
    }
    return { positions: pos, velocities: vel };
  }, [particleCount]);

  const MAX_LINES = 12000;
  const linePositions = useMemo(() => new Float32Array(MAX_LINES * 3 * 2), []);
  const lineColors = useMemo(() => new Float32Array(MAX_LINES * 3 * 2), []);

  const animState = useRef({ intensity: 0 });

  useFrame((state, delta) => {
    if (!pointsRef.current || !linesRef.current) return;
    
    const target = hovered ? 1.0 : 0.0;
    animState.current.intensity += (target - animState.current.intensity) * delta * 3.0;
    const intensity = animState.current.intensity;
    
    const posArray = pointsRef.current.geometry.attributes.position.array;
    const speed = 1.0 + intensity * 4.0;
    
    for (let i = 0; i < particleCount; i++) {
      let idx = i * 3;
      posArray[idx] += velocities[idx] * speed;
      posArray[idx+1] += velocities[idx+1] * speed;
      posArray[idx+2] += velocities[idx+2] * speed;
      
      const limit = 7;
      if (Math.abs(posArray[idx]) > limit) velocities[idx] *= -1;
      if (Math.abs(posArray[idx+1]) > limit) velocities[idx+1] *= -1;
      if (Math.abs(posArray[idx+2]) > limit) velocities[idx+2] *= -1;
    }
    
    let lineIdx = 0;
    let colorIdx = 0;
    let currentLineCount = 0;
    
    // Connect dynamic lines
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = posArray[i*3] - posArray[j*3];
        const dy = posArray[i*3+1] - posArray[j*3+1];
        const dz = posArray[i*3+2] - posArray[j*3+2];
        const distSq = dx*dx + dy*dy + dz*dz;
        
        const connectDist = maxDistance * (1.0 + intensity * 0.4);
        
        if (distSq < connectDist * connectDist) {
          if (currentLineCount >= MAX_LINES) break;
          
          const alpha = 1.0 - (Math.sqrt(distSq) / connectDist);
          
          linePositions[lineIdx++] = posArray[i*3];
          linePositions[lineIdx++] = posArray[i*3+1];
          linePositions[lineIdx++] = posArray[i*3+2];
          linePositions[lineIdx++] = posArray[j*3];
          linePositions[lineIdx++] = posArray[j*3+1];
          linePositions[lineIdx++] = posArray[j*3+2];
          
          // Blend between soft grey-blue node map and striking deep indigo data streams
          const baseR = 0.8, baseG = 0.85, baseB = 0.95; // light slate blue
          const hoverR = 0.3, hoverG = 0.27, hoverB = 0.9; // deep indigo (#4f46e5)
          
          const r = baseR + (hoverR - baseR) * intensity;
          const g = baseG + (hoverG - baseG) * intensity;
          const b = baseB + (hoverB - baseB) * intensity;
          
          // Apply opacity fading along the edges to smoothly blend into the clinical background
          const edgeAlpha = alpha * (0.3 + intensity * 0.7);
          
          lineColors[colorIdx++] = r;
          lineColors[colorIdx++] = g;
          lineColors[colorIdx++] = b;
          lineColors[colorIdx++] = r;
          lineColors[colorIdx++] = g;
          lineColors[colorIdx++] = b;
          
          currentLineCount++;
        }
      }
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    linesRef.current.geometry.attributes.position.needsUpdate = true;
    linesRef.current.geometry.attributes.color.needsUpdate = true;
    linesRef.current.geometry.setDrawRange(0, currentLineCount * 2);
    
    // Rotate constellation
    pointsRef.current.rotation.y = state.clock.elapsedTime * 0.05;
    linesRef.current.rotation.y = state.clock.elapsedTime * 0.05;
    pointsRef.current.rotation.z = state.clock.elapsedTime * 0.02;
    linesRef.current.rotation.z = state.clock.elapsedTime * 0.02;
    
    // Dolly camera
    state.camera.position.z = THREE.MathUtils.lerp(state.camera.position.z, hovered ? 5.0 : 8.0, 0.02);
  });

  return (
    <group>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={particleCount} array={positions} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial 
          size={0.08} 
          color={hovered ? "#4f46e5" : "#94a3b8"} // Deep indigo on hover, slate on idle
          transparent 
          opacity={0.8}
          depthWrite={false}
        />
      </points>
      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={MAX_LINES * 2} array={linePositions} itemSize={3} />
          <bufferAttribute attach="attributes-color" count={MAX_LINES * 2} array={lineColors} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial 
          vertexColors={true}
          transparent={true}
          opacity={0.6}
          depthWrite={false}
        />
      </lineSegments>
    </group>
  );
};

export default function NeuralNetwork() {
  const [hovered, setHovered] = useState(false);

  useEffect(() => {
    // Listen for custom trigger events from the DOM to control hover
    const handleActivate = () => setHovered(true);
    const handleDeactivate = () => setHovered(false);
    
    window.addEventListener('network-activate', handleActivate);
    window.addEventListener('network-deactivate', handleDeactivate);
    return () => {
      window.removeEventListener('network-activate', handleActivate);
      window.removeEventListener('network-deactivate', handleDeactivate);
    }
  }, []);

  return (
    <div 
      style={{ 
        position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: -10 
      }}
    >
      <Canvas camera={{ position: [0, 0, 8], fov: 60 }} gl={{ antialias: false, alpha: false }}>
        {/* Bright pristine clinical white background */}
        <color attach="background" args={["#f8fafc"]} />
        <NetworkMap hovered={hovered} />
      </Canvas>
    </div>
  );
}
