import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial, Float, Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

// An abstract 3D representation of a "neural structure" or medical sphere
const AbstractNeuralCore = () => {
  const meshRef = useRef();

  useFrame((state) => {
    if(meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.2;
      meshRef.current.rotation.x = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={1.5}>
      {/* Outer distorted wireframe (AI/Tech feeling) */}
      <Sphere ref={meshRef} args={[1.2, 64, 64]} scale={1.2}>
        <MeshDistortMaterial 
          color="#4F46E5" 
          attach="material" 
          distort={0.4} 
          speed={2} 
          roughness={0.2} 
          metalness={0.8}
          wireframe={true}
          transparent={true}
          opacity={0.6}
        />
      </Sphere>
      
      {/* Inner solid glowing core (The Brain/Data) */}
      <Sphere args={[0.9, 32, 32]}>
         <meshStandardMaterial color="#06B6D4" emissive="#06B6D4" emissiveIntensity={0.6} roughness={0.1} />
      </Sphere>
    </Float>
  );
};

// Background Particle system simulating synapses or data points
const Synapses = () => {
  const pointsRef = useRef();
  
  // Generate random points in a sphere volume using useMemo so it doesn't recalculate each render
  const particlesCount = 600;
  
  const positions = useMemo(() => {
    const pos = new Float32Array(particlesCount * 3);
    for(let i = 0; i < particlesCount; i++) {
      const radius = 2.5 + Math.random() * 3.5;
      const theta = Math.random() * 2 * Math.PI;
      const phi = Math.acos(2 * Math.random() - 1);
      
      pos[i*3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i*3+1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i*3+2] = radius * Math.cos(phi);
    }
    return pos;
  }, [particlesCount]);


  useFrame((state) => {
    if(pointsRef.current) {
      pointsRef.current.rotation.y = state.clock.elapsedTime * 0.05;
      pointsRef.current.rotation.x = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <Points ref={pointsRef} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial 
        transparent 
        color="#A78BFA" 
        size={0.06} 
        sizeAttenuation={true} 
        depthWrite={false} 
      />
    </Points>
  );
};

export default function NeuralCanvas() {
  return (
    <div style={{ width: '100%', height: '450px', position: 'relative', zIndex: 10, marginTop: '-2rem', marginBottom: '2rem' }}>
      <Canvas camera={{ position: [0, 0, 7], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffffff" />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} color="#4F46E5" />
        
        <AbstractNeuralCore />
        <Synapses />
        
        <OrbitControls 
          enableZoom={false} 
          enablePan={false}
          autoRotate={true}
          autoRotateSpeed={0.5} 
        />
      </Canvas>
    </div>
  );
}
