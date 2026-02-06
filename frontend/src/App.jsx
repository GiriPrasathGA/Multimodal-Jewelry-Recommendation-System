import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ResultsGrid from './components/ResultsGrid';
import ProductModal from './components/ProductModal';
import Toast from './components/Toast';
import AuraCursor from './components/AuraCursor';
import LiveRatesTicker from './components/LiveRatesTicker';
import { Search, Upload, Loader2, Sparkles, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// Define API Base URL (Vercel support)
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
    const [activeTab, setActiveTab] = useState('home');
    const [textQuery, setTextQuery] = useState('');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const [uploadPreview, setUploadPreview] = useState(null);
    const [handwritingMode, setHandwritingMode] = useState('sketch');
    const [selectedItem, setSelectedItem] = useState(null);
    const [toast, setToast] = useState(null);
    const [quickTags, setQuickTags] = useState([]);

    React.useEffect(() => {
        const fetchTags = async () => {
            try {
                // Add timestamp to prevent caching and force fresh random tags
                const res = await axios.get(`${API_BASE_URL}/tags?t=${Date.now()}`);
                if (res.data.tags && res.data.tags.length > 0) {
                    setQuickTags(res.data.tags);
                } else {
                    setQuickTags(['Gold', 'Diamond', 'Antique', 'Necklace', 'Engagement']);
                }
            } catch (err) {
                console.error("Failed to fetch tags", err);
                setQuickTags(['Gold', 'Diamond', 'Antique', 'Necklace', 'Engagement']);
            }
        };
        fetchTags();
    }, []);

    const handleTextSearch = async (e) => {
        e.preventDefault();
        if (!textQuery.trim()) return;
        setLoading(true);
        setResults([]); // Clear previous to show skeletons
        try {
            const res = await axios.post(`${API_BASE_URL}/search/text`, { query: textQuery, top_k: 12 });
            setResults(res.data);
            if (res.data.length > 0) setToast({ type: 'success', message: `Found ${res.data.length} exquisite items matching your vision.` });
            else setToast({ type: 'info', message: "No exact matches found. Try a different description." });
        } catch (err) { console.error("Search failed", err); } finally { setLoading(false); }
    };

    const [isScanning, setIsScanning] = useState(false);
    const [scanStep, setScanStep] = useState(0);

    const handleFileUpload = async (file, endpoint) => {
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => setUploadPreview(e.target.result);
        reader.readAsDataURL(file);

        // Start Visual DNA Scan
        setIsScanning(true);
        setResults([]);

        // Simulate AI Analysis Steps
        const steps = ["Identifying Geometry...", "Analyzing Light Refraction...", "Matching Historical Archives...", "Curating Selection..."];
        for (let i = 0; i < steps.length; i++) {
            setScanStep(i);
            await new Promise(r => setTimeout(r, 600)); // 600ms per step
        }

        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('top_k', 12);
        try {
            const res = await axios.post(`${API_BASE_URL}/search/${endpoint}`, formData);
            setResults(res.data);
            setToast({ type: 'success', message: "Visual analysis complete. Presenting curated matches." });
        } catch (err) { console.error("Upload failed", err); }
        finally {
            setLoading(false);
            setIsScanning(false);
        }
    };

    const onDrop = (e, endpoint) => {
        e.preventDefault(); e.stopPropagation(); setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0], endpoint);
    };

    const handleDrag = (e) => {
        e.preventDefault(); e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
        else if (e.type === "dragleave") setDragActive(false);
    };

    const resetSearch = () => { setResults([]); setTextQuery(""); setUploadPreview(null); };
    React.useEffect(() => { resetSearch(); }, [activeTab]);

    return (
        <div className="flex min-h-screen selection:bg-[#D4AF37]/30 selection:text-white overflow-hidden relative cursor-none">
            <AuraCursor />

            {/* Liquid Gold Atmosphere Background */}
            <div className="fixed inset-0 z-0 bg-[#050505]">
                {/* Animated Orbs - Hardware Accelerated */}
                <div className="absolute top-[-20%] left-[-10%] w-[70vw] h-[70vw] bg-[#D4AF37] rounded-full mix-blend-screen filter blur-[150px] opacity-10 animate-float will-change-transform" style={{ animationDuration: '25s' }} />
                <div className="absolute bottom-[-20%] right-[-10%] w-[60vw] h-[60vw] bg-[#9E7922] rounded-full mix-blend-screen filter blur-[150px] opacity-10 animate-float will-change-transform" style={{ animationDelay: '-5s', animationDuration: '30s' }} />
                <div className="absolute top-[40%] left-[30%] w-[40vw] h-[40vw] bg-[#F3E5AB] rounded-full mix-blend-screen filter blur-[120px] opacity-5 animate-pulse will-change-transform" style={{ animationDuration: '10s' }} />
            </div>

            <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

            <ProductModal item={selectedItem} isOpen={!!selectedItem} onClose={() => setSelectedItem(null)} />

            <AnimatePresence>
                {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
            </AnimatePresence>

            {/* Main Content Area - Shifted for sidebar */}
            <main className="lg:ml-[20rem] ml-0 flex-1 p-6 lg:p-12 pb-32 lg:pb-12 relative z-10 transition-all duration-500">

                {/* Top Header */}
                <header className="flex justify-between items-center mb-8 lg:mb-16">
                    {/* Mobile Brand Name */}
                    <div className="lg:hidden">
                        <h1 className="text-2xl font-playfair text-gold-gradient font-bold tracking-wide">Jewelry</h1>
                    </div>

                    <div className="hidden lg:flex items-center gap-2 text-sm text-[#8B949E] tracking-widest uppercase opacity-70 hover:opacity-100 transition-opacity cursor-default">
                        <span>App</span>
                        <ChevronRight size={14} />
                        <span className="text-white">{activeTab.replace(/^\w/, c => c.toUpperCase())}</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full border border-white/10 bg-white/5 flex items-center justify-center shadow-lg shadow-[#D4AF37]/5 backdrop-blur-md">
                            <Sparkles size={16} className="text-[#D4AF37]" />
                        </div>
                    </div>
                </header>

                <AnimatePresence mode="wait">
                    {activeTab === 'home' && (
                        <motion.div
                            key="home"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -30 }}
                            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                            className="flex flex-col items-start justify-center min-h-[60vh] relative"
                        >
                            <div className="mb-8 inline-flex items-center gap-3 px-4 py-2 rounded-full border border-[#D4AF37]/30 bg-[#D4AF37]/5 backdrop-blur-sm">
                                <span className="w-1.5 h-1.5 rounded-full bg-[#D4AF37] animate-pulse" />
                                <span className="text-[#D4AF37] text-xs font-bold tracking-widest uppercase">Jewelry v5.0</span>
                            </div>

                            <h1 className="text-5xl lg:text-8xl font-playfair font-bold text-transparent bg-clip-text bg-gradient-to-br from-white via-white to-white/40 mb-8 leading-tight drop-shadow-2xl">
                                Future <br />
                                <span className="text-gold-gradient italic">Holographic</span> Luxury.
                            </h1>

                            <p className="text-[#8B949E] text-xl max-w-xl font-light leading-relaxed mb-12">
                                Experience the collection through a prism of AI and light.
                            </p>

                            <button
                                onClick={() => setActiveTab('text')}
                                className="group flex items-center gap-4 px-8 py-4 bg-white text-black rounded-full font-bold transition-all hover:bg-[#D4AF37] hover:scale-105 shadow-[0_0_40px_-10px_rgba(255,255,255,0.3)] hover:shadow-[0_0_40px_-10px_#D4AF37]"
                            >
                                <span>Enter Prism</span>
                                <div className="w-8 h-8 rounded-full bg-black text-white flex items-center justify-center group-hover:bg-white group-hover:text-black transition-colors">
                                    <ChevronRight size={16} />
                                </div>
                            </button>

                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.5, duration: 0.8 }}
                            >
                                <LiveRatesTicker />
                            </motion.div>
                        </motion.div>
                    )}

                    {/* SEARCH LAYOUTS */}
                    {activeTab !== 'home' && (
                        <motion.div
                            key="search"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                        >
                            {/* INPUT SECTION */}
                            {activeTab === 'text' && (
                                <div className="mb-16 max-w-3xl">
                                    <motion.h2
                                        initial={{ opacity: 0, y: 30 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                                        className="text-4xl font-playfair text-white mb-8"
                                    >
                                        Describe your vision
                                    </motion.h2>
                                    <form onSubmit={handleTextSearch} className="relative group z-20">
                                        <div className="absolute -inset-1 bg-gradient-to-r from-[#D4AF37] to-[#F3E5AB] rounded-2xl opacity-20 group-hover:opacity-60 blur transition duration-1000"></div>
                                        <div className="relative flex items-center bg-[#0B0D12]/80 backdrop-blur-xl rounded-2xl border border-white/10 p-2 shadow-2xl">
                                            <Search className="ml-6 text-[#8B949E]" size={24} />
                                            <input
                                                type="text"
                                                value={textQuery}
                                                onChange={(e) => setTextQuery(e.target.value)}
                                                placeholder="e.g., A vintage emerald necklace..."
                                                className="w-full bg-transparent text-white p-4 text-2xl focus:outline-none placeholder:text-[#8B949E]/40 font-playfair italic tracking-wide"
                                            />
                                            <button
                                                type="submit"
                                                disabled={loading}
                                                className="px-8 py-3 bg-[#D4AF37] text-black rounded-xl font-bold hover:bg-[#F3E5AB] transition-colors"
                                            >
                                                {loading ? <Loader2 className="animate-spin" /> : "Find"}
                                            </button>
                                        </div>
                                    </form>

                                    {/* Quick Filters - Horizontal Scroll Rail */}
                                    <div className="relative w-full max-w-4xl mx-auto mb-8 -mx-6 lg:mx-auto">
                                        <div
                                            className="flex gap-4 overflow-x-auto pb-4 pt-2 px-6 lg:px-12 no-scrollbar scroll-smooth"
                                            style={{ maskImage: 'linear-gradient(to right, transparent, black 10%, black 90%, transparent)', WebkitMaskImage: 'linear-gradient(to right, transparent, black 10%, black 90%, transparent)' }}
                                        >
                                            {quickTags.map((tag) => (
                                                <button
                                                    key={tag}
                                                    onClick={() => { setTextQuery(tag); }}
                                                    className="group relative flex-shrink-0 flex items-center gap-3 px-6 py-3 rounded-full border border-white/10 bg-white/5 hover:bg-[#D4AF37]/10 hover:border-[#D4AF37]/40 transition-all duration-300 whitespace-nowrap"
                                                >
                                                    <span className="w-1.5 h-1.5 rounded-full bg-[#D4AF37]/40 group-hover:bg-[#D4AF37] group-hover:shadow-[0_0_8px_#D4AF37] transition-all duration-300" />
                                                    <span className="text-[#8B949E] group-hover:text-white font-playfair italic text-lg tracking-wide capitalize transition-colors">
                                                        {tag}
                                                    </span>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {(activeTab === 'image' || activeTab === 'handwriting') && (
                                <div className="mb-16 grid grid-cols-1 lg:grid-cols-12 gap-8">
                                    <div className="col-span-1 lg:col-span-8">
                                        <h2 className="text-3xl font-playfair text-white mb-6">
                                            {activeTab === 'image' ? "Visual Search" : "Creative Input"}
                                        </h2>
                                        <div
                                            className={`relative h-64 glass-panel rounded-3xl border-dashed border-2 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 group overflow-hidden ${dragActive ? 'border-[#D4AF37] bg-[#D4AF37]/5' : 'border-white/10 hover:border-[#D4AF37]/50'
                                                }`}
                                            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag}
                                            onDrop={(e) => onDrop(e, activeTab === 'image' ? 'image' : handwritingMode)}
                                        >

                                            {/* SCANNER OVERLAY */}
                                            {isScanning && (
                                                <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center">
                                                    <div className="w-full h-1 bg-[#D4AF37] shadow-[0_0_20px_#D4AF37] absolute top-0 animate-scan-down"></div>
                                                    <div className="text-[#D4AF37] font-mono text-sm tracking-[0.2em] mb-4 animate-pulse">
                                                        {["Identifying Geometry...", "Analyzing Light Refraction...", "Matching Historical Archives...", "Curating Selection..."][scanStep]}
                                                    </div>

                                                    {/* Holographic Grid */}
                                                    <div className="absolute inset-0 bg-[linear-gradient(rgba(212,175,55,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(212,175,55,0.1)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none"></div>
                                                </div>
                                            )}

                                            <input type="file" id="file" className="hidden" onChange={(e) => handleFileUpload(e.target.files[0], activeTab === 'image' ? 'image' : handwritingMode)} />

                                            {!uploadPreview ? (
                                                <label htmlFor="file" className="text-center cursor-pointer p-10 w-full h-full flex flex-col items-center justify-center z-20">
                                                    <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500 shadow-xl border border-white/5">
                                                        <Upload size={32} className="text-[#D4AF37]" />
                                                    </div>
                                                    <p className="text-xl font-playfair text-white mb-2">Drop your image here</p>
                                                    <p className="text-[#8B949E] text-sm tracking-widest uppercase">or click to browse</p>
                                                </label>
                                            ) : (
                                                <div className="relative w-full h-full p-4 group">
                                                    <img src={uploadPreview} className="w-full h-full object-contain rounded-xl relative z-10" />
                                                    <div className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-sm rounded-3xl z-20">
                                                        <button onClick={resetSearch} className="px-6 py-2 border border-white rounded-full text-white hover:bg-white hover:text-black transition-colors">Replace</button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {activeTab === 'handwriting' && (
                                        <div className="col-span-1 lg:col-span-4 glass-panel rounded-3xl p-6">
                                            <h3 className="text-[#D4AF37] font-bold text-sm uppercase tracking-widest mb-6 border-b border-white/10 pb-4">Mode Selection</h3>
                                            <div className="space-y-3">
                                                {['sketch', 'handwriting'].map((mode) => (
                                                    <div
                                                        key={mode}
                                                        onClick={() => setHandwritingMode(mode)}
                                                        className={`p-4 rounded-xl cursor-pointer border transition-all duration-300 ${handwritingMode === mode
                                                            ? 'bg-[#D4AF37]/20 border-[#D4AF37] shadow-[0_0_20px_-5px_#D4AF37]'
                                                            : 'bg-transparent border-white/10 hover:bg-white/5'
                                                            }`}
                                                    >
                                                        <div className="flex items-center justify-between mb-2">
                                                            <span className="font-playfair text-lg capitalize">{mode === 'sketch' ? 'Artistic Sketch' : 'Handwritten Note'}</span>
                                                            <div className={`w-4 h-4 rounded-full border flex items-center justify-center ${handwritingMode === mode ? 'border-[#D4AF37]' : 'border-[#8B949E]'}`}>
                                                                {handwritingMode === mode && <div className="w-2 h-2 bg-[#D4AF37] rounded-full" />}
                                                            </div>
                                                        </div>
                                                        <p className="text-xs text-[#8B949E] leading-relaxed">
                                                            {mode === 'sketch' ? 'Use for rough drawings or shapes.' : 'Use for written descriptions on paper.'}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* RESULTS */}
                            <ResultsGrid
                                loading={loading}
                                results={results}
                                onCardClick={(item) => setSelectedItem(item)}
                            />
                        </motion.div>
                    )}
                </AnimatePresence>

                <ProductModal
                    item={selectedItem}
                    isOpen={!!selectedItem}
                    onClose={() => setSelectedItem(null)}
                />

            </main>
        </div>
    );
}

export default App;
