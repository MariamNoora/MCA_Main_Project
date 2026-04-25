import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, Activity, ArrowRight, Microscope, Target, Fingerprint, Database } from 'lucide-react';
import NeuralNetwork from '../components/NeuralNetwork'; // New background component

const Landing = () => {

  const triggerHover = (active) => {
    const event = new CustomEvent(active ? 'network-activate' : 'network-deactivate');
    window.dispatchEvent(event);
  };

  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden' }}>
      {/* Dynamic 3D Neural Constellation */}
      <NeuralNetwork />

      {/* Hero Content Overlay */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '90vh', padding: '0 2rem', paddingTop: '80px', position: 'relative', zIndex: 10 }}>
        
        <div 
          className="cyber-hero-panel animate-fly-in"
          onMouseEnter={() => triggerHover(true)}
          onMouseLeave={() => triggerHover(false)}
        >
          <div className="status-badge">System Online: Neural Core Active</div>
          <h1 className="hero-title">
            DEEP<span className="accent-text">SPECTRUM</span> AI
          </h1>
          <p className="hero-subtitle">
            Enter the architecture of next-generation diagnostic intelligence.
            Mapping microscopic structural variances via deeply layered 3D-ResNet analytics.
          </p>
          
          <div className="button-group">
             <Link to="/detection" className="cyber-btn primary-cyber" style={{ textDecoration: 'none' }}>
                <Activity size={20} /> Initialize Diagnostic Flow
             </Link>
             <a href="#architecture" className="cyber-btn secondary-cyber" style={{ textDecoration: 'none' }}>
                <Database size={20} /> Architecture Specs
             </a>
          </div>
        </div>
      </div>

      {/* Core Capabilities Section */}
      <div id="architecture" className="cyber-section animate-fly-in delay-200">
         <h2 style={{ textAlign: 'center', color: 'var(--text-main)', marginBottom: '4rem', fontSize: '2.5rem', letterSpacing: '4px', textTransform: 'uppercase' }}>Diagnostic Parameters</h2>
         
         <div className="cyber-grid">
            <div className="cyber-card">
               <Fingerprint size={48} className="card-icon" />
               <h3>Morphological Detection</h3>
               <p>Our algorithms sift through thousands of structural markers to pinpoint microscopic tissue variances indicative of ASD.</p>
            </div>
            
            <div className="cyber-card">
               <Microscope size={48} className="card-icon" />
               <h3>3D Grad-CAM</h3>
               <p>True interpretability. Examine visually mapped heat signatures over the original .nii scan to see exactly how the AI reasoned.</p>
            </div>
            
            <div className="cyber-card">
               <Target size={48} className="card-icon" />
               <h3>Severity Bracket Triage</h3>
               <p>Automatically segments detected cases into functional stages (Mild, Moderate, Severe) to expedite clinical triaging protocols.</p>
            </div>
         </div>
      </div>
      
      {/* Informational Stages Section */}
      <div id="stages" className="cyber-section animate-fly-in delay-300" style={{ paddingTop: '2rem' }}>
        <div className="cyber-card" style={{ padding: '4rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h2 style={{ color: 'var(--text-main)', fontSize: '2.5rem', marginBottom: '1rem', letterSpacing: '-1px' }}>Clinical Progression Engine</h2>
            <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto', fontSize: '1.1rem' }}>How our DEEP SPECTRUM architecture maps structural markers to actionable patient staging and severity brackets.</p>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
            
            {/* Stage 1 */}
            <div style={{ display: 'flex', gap: '2.5rem', alignItems: 'flex-start', position: 'relative' }}>
              <div style={{ position: 'absolute', left: '40px', top: '90px', bottom: '-40px', width: '2px', background: 'linear-gradient(to bottom, #10B981, #F59E0B)', opacity: 0.3 }}></div>
              <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)', color: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 10px 25px rgba(16,185,129,0.3)' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 800 }}>I</span>
                <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Mild</span>
              </div>
              <div>
                <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem', color: '#059669' }}>Early / Subtle Features</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', lineHeight: '1.6' }}>
                  Subtle microstructural differences detected by the AI. Individuals are often highly functional but may face minor social or communication challenges. Associated with confidence probabilities between 50-65%.
                </p>
              </div>
            </div>

            {/* Stage 2 */}
            <div style={{ display: 'flex', gap: '2.5rem', alignItems: 'flex-start', position: 'relative' }}>
              <div style={{ position: 'absolute', left: '40px', top: '90px', bottom: '-40px', width: '2px', background: 'linear-gradient(to bottom, #F59E0B, #EF4444)', opacity: 0.3 }}></div>
              <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)', color: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 10px 25px rgba(245,158,11,0.3)' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 800 }}>II</span>
                <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Mod</span>
              </div>
              <div>
                <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem', color: '#D97706' }}>Moderate Structural Markers</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', lineHeight: '1.6' }}>
                  Distinct morphological markers present in the MRI cortex. Indicates an increased need for support in daily activities and pronounced communication differences. Probabilities range from 65-85%.
                </p>
              </div>
            </div>

            {/* Stage 3 */}
            <div style={{ display: 'flex', gap: '2.5rem', alignItems: 'flex-start' }}>
              <div style={{ width: '80px', height: '80px', borderRadius: '24px', background: 'linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)', color: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flexShrink: 0, boxShadow: '0 10px 25px rgba(239,68,68,0.3)' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 800 }}>III</span>
                <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Severe</span>
              </div>
              <div>
                <h3 style={{ fontSize: '1.5rem', marginBottom: '0.75rem', color: '#B91C1C' }}>Advanced / Severe Features</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', lineHeight: '1.6' }}>
                  Extensive neuro-structural variance identified by the 3D ResNet. Indicates individuals requiring substantial support, often non-verbal or possessing significant behavioral adaptations. Model threshold &gt;85% certainty.
                </p>
              </div>
            </div>

          </div>
        </div>
      </div>
      
    </div>
  );
};

export default Landing;
