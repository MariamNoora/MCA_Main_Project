import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

const CinematicBrain = ({ hovered, pointsData }) => {
  const pointsRef = useRef();
  const particlesCount = pointsData.length / 3;
  
  // Base particles derived perfectly from NiLearn fsaverage anatomical dataset
  const { basePositions, directions, colors } = useMemo(() => {
    const base = new Float32Array(pointsData);
    const dirs = new Float32Array(particlesCount * 3);
    const cols = new Float32Array(particlesCount * 3);
    
    // High-tech Cyber Colors (Cyan, Indigo, Magenta)
    const colorA = new THREE.Color("#06B6D4"); // Cyan
    const colorB = new THREE.Color("#6366F1"); // Indigo
    const colorC = new THREE.Color("#d946ef"); // Fuchsia
    
    for (let i = 0; i < particlesCount; i++) {
      let x = base[i*3];
      let y = base[i*3+1];
      let z = base[i*3+2];
      
      // Explosion outward vectors
      const length = Math.sqrt(x*x + y*y + z*z) || 1;
      dirs[i*3] = (x / length) * (0.8 + Math.random() * 0.8);
      dirs[i*3+1] = (y / length) * (0.8 + Math.random() * 0.8);
      dirs[i*3+2] = (z / length) * (0.8 + Math.random() * 0.8);
      
      // Render crisp tech edge gradients
      const rMix = Math.random();
      const col = rMix > 0.6 ? colorA : (rMix > 0.2 ? colorB : colorC);
      cols[i*3] = col.r;
      cols[i*3+1] = col.g;
      cols[i*3+2] = col.b;
    }
    
    return { basePositions: base, directions: dirs, colors: cols };
  }, [pointsData, particlesCount]);

  const currentPositions = useMemo(() => new Float32Array(basePositions), [basePositions]);
  const currentColors = useMemo(() => new Float32Array(colors), [colors]);
  
  const animState = useRef({ explodeFactor: 0 });
  const highlightColor = useMemo(() => new THREE.Color("#22d3ee"), []);

  useFrame((state, delta) => {
    if(!pointsRef.current) return;
    
    const targetExplode = hovered ? 1.0 : 0.0;
    animState.current.explodeFactor += (targetExplode - animState.current.explodeFactor) * delta * 4.0;
    const factor = animState.current.explodeFactor;
    
    // Slow sweeping rotation
    pointsRef.current.rotation.y = state.clock.elapsedTime * (0.1 + factor * 0.2);
    pointsRef.current.position.y = Math.sin(state.clock.elapsedTime * 1.5) * 0.05;
    
    const positions = pointsRef.current.geometry.attributes.position.array;
    const colorArray = pointsRef.current.geometry.attributes.color.array;
    
    for(let j = 0; j < particlesCount; j++) {
        const idx = j * 3;
        // Mild explosion constrained slightly
        positions[idx]   = basePositions[idx]   + directions[idx]   * factor * 1.2; 
        positions[idx+1] = basePositions[idx+1] + directions[idx+1] * factor * 1.2; 
        positions[idx+2] = basePositions[idx+2] + directions[idx+2] * factor * 1.2; 
        
        // Tech glowing transition on explode
        const idleR = colors[idx];
        const idleG = colors[idx+1];
        const idleB = colors[idx+2];
        
        colorArray[idx]   = idleR + (highlightColor.r - idleR) * factor;
        colorArray[idx+1] = idleG + (highlightColor.g - idleG) * factor;
        colorArray[idx+2] = idleB + (highlightColor.b - idleB) * factor;
    }
    
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
    pointsRef.current.geometry.attributes.color.needsUpdate = true;
    pointsRef.current.material.size = 0.025 + (factor * 0.015);
  });

  return (
    <points ref={pointsRef} frustumCulled={false}>
      <bufferGeometry attach="geometry">
        <bufferAttribute attach="attributes-position" count={particlesCount} array={currentPositions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={particlesCount} array={currentColors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial 
        attach="material"
        transparent={true} 
        vertexColors={true}
        size={0.025} 
        sizeAttenuation={true} 
        depthWrite={false} 
        blending={THREE.AdditiveBlending}
        opacity={0.8}
      />
    </points>
  );
};

export default function NeuralBrain() {
  const [internalHover, setInternalHover] = useState(false);
  const [brainData, setBrainData] = useState(null);

  useEffect(() => {
    let mounted = true;
    
    fetch('/brain_points.json')
      .then(res => res.json())
      .then(data => {
        if (mounted) {
          setBrainData(data);
        }
      })
      .catch(err => console.error("Error loading anatomically accurate brain points", err));
      
    return () => { mounted = false; };
  }, []);

  return (
    <div 
      style={{ 
        width: '100%', height: '100%', 
        position: 'absolute', top: 0, left: 0 
      }}
      onMouseEnter={() => setInternalHover(true)}
      onMouseLeave={() => setInternalHover(false)}
      className="neural-brain-container"
    >
      <Canvas camera={{ position: [0, 0, 5.5], fov: 50 }}>
        <ambientLight intensity={1} />
        {brainData && <CinematicBrain hovered={internalHover} pointsData={brainData} />}
        <OrbitControls enableZoom={false} enablePan={false} autoRotate={false} />
      </Canvas>
    </div>
  );
}
