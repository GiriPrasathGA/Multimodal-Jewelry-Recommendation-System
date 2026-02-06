import React, { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Heart, MessageSquare, ShieldCheck, Truck, Sparkles, ZoomIn } from 'lucide-react';

const ProductModal = ({ item, isOpen, onClose }) => {
    const [loupe, setLoupe] = useState({ show: false, x: 0, y: 0 });

    const handleMouseMove = (e) => {
        const { left, top, width, height } = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - left) / width) * 100;
        const y = ((e.clientY - top) / height) * 100;
        setLoupe({ show: true, x, y });
    };

    // Safe price calculation
    const price = useMemo(() => {
        if (!item) return "₹0";
        const seed = item.id || 0;
        // Simple pseudo-random using ID
        const randomPrice = Math.floor((Math.sin(seed) * 0.5 + 0.5) * (1200000 - 250000) + 250000);
        return randomPrice.toLocaleString('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        });
    }, [item]);

    // Only render backdrop if we have an item and it's open
    // BUT we must return the AnimatePresence component always for exit animations to work if we were controlling it from outside
    // However, here we control internal AnimatePresence.

    // If item is null, we can't render the modal content.
    const showModal = isOpen && !!item;

    return (
        <AnimatePresence>
            {showModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-0 md:p-4 sm:p-8">
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="absolute inset-0 bg-[#0B0D12]/80 backdrop-blur-md transition-opacity"
                    />

                    {/* Modal Content */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 40 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 40 }}
                        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                        className="relative w-full max-w-6xl h-full md:h-[85vh] bg-[#161B22] border-x-0 border-y-0 md:border md:border-white/10 md:rounded-3xl overflow-hidden shadow-2xl flex flex-col md:flex-row shadow-[#D4AF37]/10"
                    >
                        {/* Close Button */}
                        <button
                            onClick={onClose}
                            className="absolute top-6 right-6 z-20 p-2 rounded-full bg-black/40 text-white hover:bg-white hover:text-black transition-all duration-300 border border-white/10"
                        >
                            <X size={24} />
                        </button>

                        {/* Left: Image (The Jeweler's Loupe) */}
                        <div
                            className="w-full md:w-[55%] h-[45vh] md:h-full relative overflow-hidden bg-black group cursor-none"
                            onMouseMove={handleMouseMove}
                            onMouseLeave={() => setLoupe({ ...loupe, show: false })}
                        >
                            <img
                                src={`data:image/jpeg;base64,${item.image_base64}`}
                                alt={item.category}
                                className="w-full h-full object-cover opacity-90 transition-transform duration-1000"
                            />

                            {/* The Loupe Lens */}
                            <AnimatePresence>
                                {loupe.show && (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.8 }}
                                        className="absolute w-48 h-48 rounded-full border-2 border-[#D4AF37]/50 shadow-[0_0_30px_rgba(0,0,0,0.5),inset_0_0_20px_rgba(212,175,55,0.2)] bg-no-repeat pointer-events-none z-10 overflow-hidden"
                                        style={{
                                            left: `${loupe.x}%`,
                                            top: `${loupe.y}%`,
                                            transform: 'translate(-50%, -50%)',
                                            backgroundImage: `url(data:image/jpeg;base64,${item.image_base64})`,
                                            backgroundPosition: `${loupe.x}% ${loupe.y}%`,
                                            backgroundSize: '200%', // 2x Zoom
                                            backgroundColor: '#000'
                                        }}
                                    >
                                        {/* Lens Reflection/Glare */}
                                        <div className="absolute inset-0 bg-gradient-to-tr from-white/10 to-transparent rounded-full" />
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            <div className="absolute inset-0 bg-gradient-to-t from-[#161B22] via-transparent to-transparent md:bg-gradient-to-r pointer-events-none" />

                            {/* Floating Price Tag on Image */}
                            <div className="absolute bottom-8 left-8 bg-black/60 backdrop-blur-xl px-6 py-3 rounded-full border border-[#D4AF37]/30 text-white font-playfair text-2xl md:text-3xl font-bold tracking-wide shadow-2xl pointer-events-none">
                                {price}
                            </div>

                            {/* Instruction Hint */}
                            <div className={`absolute top-8 left-8 flex items-center gap-2 text-xs text-white/50 uppercase tracking-widest transition-opacity duration-300 ${loupe.show ? 'opacity-0' : 'opacity-100'}`}>
                                <ZoomIn size={14} /> Hover to Inspect
                            </div>
                        </div>

                        {/* Right: Details */}
                        <div className="w-full md:w-[45%] p-8 md:p-12 flex flex-col h-full overflow-y-auto custom-scrollbar relative">
                            {/* Watermark */}
                            <div className="absolute top-10 right-10 text-[120px] font-playfair text-white/5 pointer-events-none leading-none opacity-50">
                                JL
                            </div>

                            <div className="mb-auto relative z-10">
                                <div className="flex items-center gap-3 mb-6">
                                    <span className="px-3 py-1 rounded-full border border-[#D4AF37]/30 bg-[#D4AF37]/5 text-[#D4AF37] text-xs font-bold tracking-widest uppercase">
                                        JewelUX Exclusive
                                    </span>
                                    <span className="text-[#8B949E] text-xs tracking-widest uppercase">
                                        Ref: #{item.id || 'N/A'}
                                    </span>
                                </div>

                                <h2 className="text-4xl md:text-5xl font-playfair text-white mb-6 leading-[1.1]">
                                    {item.category}
                                </h2>

                                <div className="flex items-center gap-6 mb-8 text-sm text-[#8B949E] font-medium tracking-wide">
                                    <div className="flex items-center gap-2">
                                        <ShieldCheck size={16} className="text-emerald-400" />
                                        Certified Authentic
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <Truck size={16} className="text-blue-400" />
                                        Insured Delivery
                                    </div>
                                </div>

                                <div className="h-px w-full bg-white/10 mb-8" />

                                <h3 className="text-sm font-bold uppercase tracking-widest text-[#D4AF37] mb-4">Description</h3>
                                <p className="text-[#E0E0E0]/80 text-lg font-light leading-relaxed mb-10">
                                    {item.description}
                                </p>

                                <div className="mb-8 p-6 rounded-2xl bg-gradient-to-br from-[#D4AF37]/10 to-transparent border border-[#D4AF37]/20 relative overflow-hidden group">
                                    <div className="absolute top-0 right-0 p-3 opacity-20 group-hover:opacity-40 transition-opacity">
                                        <Sparkles size={48} />
                                    </div>
                                    <h3 className="text-[#D4AF37] font-bold text-sm uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <Sparkles size={14} /> JewelGPT Insights
                                    </h3>
                                    <div className="flex flex-wrap gap-2 mb-4">
                                        {((item.id || 0) % 2 === 0 ? ["Red Carpet Ready", "Heirloom Quality"] : ["Modern Minimalist", "Avante-Garde"]).map(tag => (
                                            <span key={tag} className="px-3 py-1 rounded-full bg-[#D4AF37]/10 text-[#D4AF37] text-xs font-medium border border-[#D4AF37]/20">
                                                {tag}
                                            </span>
                                        ))}
                                        <span className="px-3 py-1 rounded-full bg-white/5 text-[#8B949E] text-xs font-medium border border-white/10">
                                            Rarity: {(item.id || 0) % 3 === 0 ? "Ultra Rare" : "Limited Edition"}
                                        </span>
                                    </div>
                                    <p className="text-sm text-white/70 italic leading-relaxed">
                                        "{(item.id || 0) % 2 === 0
                                            ? "This piece exhibits a masterful balance of classical geometry and modern brilliance. A timeless investment."
                                            : "A bold statement piece that defies convention. Perfect for the collector who values distinct artistic expression."}"
                                    </p>
                                </div>

                                <div className="grid grid-cols-2 gap-4 mb-8">
                                    <div className="p-4 rounded-2xl bg-white/5 border border-white/5 hover:border-[#D4AF37]/30 transition-colors">
                                        <span className="block text-[#8B949E] text-xs uppercase tracking-widest mb-1">Visual Match</span>
                                        <span className="text-3xl font-playfair text-[#D4AF37]">{Math.round(item.score * 100)}%</span>
                                    </div>
                                    <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                                        <span className="block text-[#8B949E] text-xs uppercase tracking-widest mb-1">Gem Purity</span>
                                        <span className="text-3xl font-playfair text-white">VVS1</span>
                                    </div>
                                </div>
                            </div>

                            {/* Action Buttons */}
                            <div className="flex flex-col gap-4 mt-8 pt-8 border-t border-white/10 relative z-10">
                                <button className="w-full py-4 bg-[#D4AF37] text-black text-sm font-bold uppercase tracking-[0.2em] rounded-xl hover:bg-[#F3E5AB] transition-colors flex items-center justify-center gap-3 shadow-[0_0_30px_-10px_#D4AF37]">
                                    <MessageSquare size={18} />
                                    Request Consultation
                                </button>
                                <button className="w-full py-4 bg-transparent border border-white/20 text-white text-sm font-bold uppercase tracking-[0.2em] rounded-xl hover:bg-white hover:text-black transition-all flex items-center justify-center gap-3">
                                    <Heart size={18} />
                                    Add to Wishlist
                                </button>
                            </div>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
};

export default ProductModal;
