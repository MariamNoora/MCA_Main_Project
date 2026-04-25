import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, AlertCircle, FileText, Loader2, Info } from 'lucide-react';

const Detection = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setError('');
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a valid .nii or .nii.gz file.");
      return;
    }

    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/diagnose', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during diagnosis.");
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (verdict) => {
    if (verdict.includes('ASD Positive')) return '#EF4444'; // Red
    if (verdict.includes('Neurotypical')) return '#10B981'; // Green
    return '#F59E0B'; // Orange for OOD
  };

  return (
    <div className="detection-container animate-fly-in">
      <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
        <h2 style={{ marginBottom: '1rem' }}>Volumetric <span className="text-gradient">Diagnostics</span> Engine</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.2rem', maxWidth: '600px', margin: '0 auto' }}>Upload T1-weighted MRI volumes for instant, explainable AI analysis.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem', maxWidth: '1000px', margin: '0 auto' }}>
        
        {/* Upload Section - Transformed to look stunning */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', padding: '4rem 3rem' }}>
          
          <div style={{ 
            width: '100%', 
            maxWidth: '600px',
            border: `2px dashed ${file ? 'var(--primary)' : 'rgba(79, 70, 229, 0.3)'}`, 
            borderRadius: '24px', 
            padding: '4rem 2rem', 
            background: file ? 'rgba(79, 70, 229, 0.05)' : 'rgba(255, 255, 255, 0.5)',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            boxShadow: file ? '0 0 20px rgba(79, 70, 229, 0.1)' : 'none'
          }}
          onClick={() => document.getElementById('fileUpload').click()}
          className="upload-box"
          >
            <input 
              id="fileUpload" 
              type="file" 
              accept=".nii,.nii.gz" 
              onChange={handleFileChange} 
              style={{ display: 'none' }} 
            />
            {file ? (
              <div className="animate-fly-in">
                <FileText size={56} color="var(--primary)" style={{ margin: '0 auto 1.5rem', filter: 'drop-shadow(0 4px 6px rgba(79,70,229,0.3))' }} />
                <h4 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.5rem' }}>{file.name}</h4>
                <p style={{ color: 'var(--text-muted)', fontSize: '1rem', fontWeight: 500 }}>Ready for volumetric analysis</p>
              </div>
            ) : (
              <div>
                <UploadCloud size={56} color="var(--primary-light)" style={{ margin: '0 auto 1.5rem' }} />
                <h4 style={{ color: 'var(--text-main)', marginBottom: '0.5rem', fontSize: '1.5rem' }}>Drop MRI Scan Here</h4>
                <p style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>Click to browse .nii or .nii.gz files</p>
              </div>
            )}
          </div>

          <button 
            className="btn btn-primary" 
            style={{ marginTop: '3rem', width: '100%', maxWidth: '300px', padding: '1.25rem' }}
            onClick={handleUpload}
            disabled={!file || loading}
          >
            {loading ? <><Loader2 className="animate-spin" size={24} /> Processing Volume...</> : 'Initiate Analysis Cycle'}
          </button>

          {error && (
            <div className="animate-fly-in" style={{ marginTop: '2rem', padding: '1rem 1.5rem', background: 'rgba(244,63,94,0.1)', color: 'var(--accent)', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '0.75rem', fontWeight: 500 }}>
              <AlertCircle size={24} /> {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="glass-card animate-fly-in" style={{ padding: '0', overflow: 'hidden', border: `1px solid ${getStatusColor(result.verdict)}40` }}>
            
            <div style={{ 
              padding: '3rem', 
              borderBottom: '1px solid rgba(0,0,0,0.05)', 
              background: result.is_ood ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.05), transparent)' 
                        : (result.verdict.includes('ASD') ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.05), transparent)' 
                        : 'linear-gradient(135deg, rgba(16, 185, 129, 0.05), transparent)') 
            }}>
              
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '2rem', marginBottom: '2rem' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', margin: 0 }}>
                  Diagnostic Consensus
                  <span style={{ 
                    fontSize: '1.125rem', 
                    padding: '0.5rem 1.5rem', 
                    borderRadius: '999px', 
                    border: `1px solid ${getStatusColor(result.verdict)}`,
                    color: getStatusColor(result.verdict),
                    background: `${getStatusColor(result.verdict)}15`,
                    fontWeight: 700,
                    boxShadow: `0 4px 15px ${getStatusColor(result.verdict)}25`
                  }}>
                    {result.verdict}
                  </span>
                </h3>
                <div style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                  Processed in <span style={{ color: 'var(--text-main)' }}>{result.inference_time}</span>
                </div>
              </div>
              
              {result.is_ood ? (
                 <div style={{ background: '#FFF9C4', border: '1px solid #FBC02D', padding: '1.5rem', borderRadius: '16px', color: '#B71C1C', display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                   <Info size={28} style={{ flexShrink: 0, marginTop: '2px' }}/> 
                   <div style={{ fontSize: '1.1rem', lineHeight: 1.6 }}>
                     <strong>Out-of-Distribution Warning:</strong> The AI model detected that this scan is highly anomalous. The structural density or predictive entropy indicates this is likely a non-brain image, a pathological anomaly (like a severe tumor), or a heavily degraded scan. Analysis halted for clinical safety.
                   </div>
                 </div>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem' }}>
                  <div style={{ background: '#FFFFFF', padding: '2rem', borderRadius: '20px', boxShadow: 'var(--shadow-sm)', border: '1px solid var(--card-border)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem', fontWeight: 600 }}>Confidence</div>
                    <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--primary)' }}>{result.confidence}</div>
                  </div>
                  <div style={{ background: '#FFFFFF', padding: '2rem', borderRadius: '20px', boxShadow: 'var(--shadow-sm)', border: '1px solid var(--card-border)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem', fontWeight: 600 }}>Severity Level</div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-main)' }}>{result.severity}</div>
                  </div>
                  <div style={{ background: '#FFFFFF', padding: '2rem', borderRadius: '20px', boxShadow: 'var(--shadow-sm)', border: '1px solid var(--card-border)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem', fontWeight: 600 }}>Progression Stage</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-main)', lineHeight: 1.2 }}>{result.stage}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Grad-CAM Viewer */}
            <div style={{ padding: '3rem' }}>
              <h4 style={{ marginBottom: '2rem', fontSize: '1.5rem' }}>Explainability (Grad-CAM) Visual Matrix</h4>
              
              <div style={{ display: 'flex', gap: '3rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <div style={{ position: 'relative', flex: '1', minWidth: '300px', background: '#000', borderRadius: '24px', overflow: 'hidden', padding: '1.5rem', display: 'flex', justifyContent: 'center', boxShadow: '0 20px 40px -10px rgba(0,0,0,0.3)' }}>
                  {result.gradcam_image && (
                    <img 
                      src={result.gradcam_image} 
                      alt="MRI Heatmap" 
                      style={{ maxWidth: '100%', height: 'auto', borderRadius: '12px' }} 
                    />
                  )}
                  {/* Decorative corner markers */}
                  <div style={{ position: 'absolute', top: 0, left: 0, width: '20px', height: '20px', borderTop: '2px solid var(--secondary)', borderLeft: '2px solid var(--secondary)', margin: '1rem' }}></div>
                  <div style={{ position: 'absolute', bottom: 0, right: 0, width: '20px', height: '20px', borderBottom: '2px solid var(--secondary)', borderRight: '2px solid var(--secondary)', margin: '1rem' }}></div>
                </div>
                
                {!result.is_ood && (
                  <div style={{ flex: '1.5', minWidth: '350px' }}>
                    <h5 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: 'var(--primary)' }}>Interpretive Insights</h5>
                    <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', lineHeight: 1.7, marginBottom: '2rem' }}>
                      The heatmap overlay reveals the localized spatial volumes that most heavily influenced the AI's diagnostic tensor. 
                      <strong> Warmer colors (Red/Yellow) </strong> directly indicate high-attention structural anomalies corresponding with the provided prediction.
                    </p>
                    
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-muted)' }}>Low Activation</span>
                        <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--primary)' }}>High Activation</span>
                      </div>
                      <div style={{ height: '12px', background: 'linear-gradient(to right, rgba(0,0,255,0.7), rgba(0,255,255,0.7), rgba(0,255,0,0.7), rgba(255,255,0,0.7), rgba(255,0,0,0.7))', borderRadius: '6px', boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)' }}></div>
                    </div>
                  </div>
                )}
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
};

export default Detection;
