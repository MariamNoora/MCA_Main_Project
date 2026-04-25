import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';
import Landing from './pages/Landing';
import Detection from './pages/Detection';

function Navigation() {
  const location = useLocation();
  
  return (
    <header className="nav-header">
      <Link to="/" className="nav-logo" style={{ textDecoration: 'none' }}>
        <Activity size={28} color="var(--primary)" />
        <span style={{ fontWeight: 800 }}>DEEP</span><span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>SPECTRUM</span>
      </Link>
      <nav className="nav-links">
        <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`} style={{ textDecoration: 'none' }}>
          Overview
        </Link>
        <Link to="/detection" className={`cyber-btn primary-cyber`} style={{ padding: '0.6rem 1.4rem', fontSize: '0.9rem', textDecoration: 'none' }}>
          Diagnostic Portal
        </Link>
      </nav>
    </header>
  );
}

function App() {
  return (
    <Router>
      <Navigation />
      <div className="page-wrapper container">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/detection" element={<Detection />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
