import React from 'react';
import { Home, Type, Image as ImageIcon, PenTool } from 'lucide-react';
import { motion } from 'framer-motion';

const Navbar = ({ activeTab, setActiveTab }) => {

    const navItems = [
        { id: 'home', label: 'Home', icon: Home },
        { id: 'text', label: 'Text Search', icon: Type },
        { id: 'image', label: 'Image Search', icon: ImageIcon },
        { id: 'handwriting', label: 'Handwriting', icon: PenTool },
    ];

    return (

        <div className="fixed bottom-4 left-4 right-4 h-20 lg:h-auto lg:left-6 lg:top-6 lg:bottom-6 lg:w-72 bg-[#0B0D12]/80 lg:bg-[#0B0D12]/60 backdrop-blur-2xl border border-white/10 shadow-2xl rounded-2xl lg:p-6 px-6 flex flex-row lg:flex-col items-center lg:items-stretch justify-between z-50 transition-all duration-300">

            {/* Brand (Desktop Only) */}
            <div className="hidden lg:block mb-12 px-2">
                <h1 className="text-3xl font-playfair text-gold-gradient font-bold tracking-wide">Jewelry</h1>
                <p className="text-[#8B949E] text-xs tracking-[0.2em] uppercase mt-1">Premium Collection</p>
            </div>

            {/* Menu */}
            <div className="flex flex-row lg:flex-col w-full lg:w-auto justify-between lg:justify-start lg:space-y-3">
                <label className="hidden lg:block text-xs font-bold text-[#8B949E]/50 uppercase tracking-widest px-4 mb-4">
                    Discover
                </label>

                {navItems.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`relative flex flex-col lg:flex-row items-center lg:gap-4 p-2 lg:px-4 lg:py-3.5 rounded-xl transition-colors duration-300 group overflow-hidden ${activeTab === item.id
                            ? 'text-white'
                            : 'text-[#8B949E] hover:text-white hover:bg-white/5'
                            }`}
                    >
                        {/* Active Background Sliding Pill */}
                        {activeTab === item.id && (
                            <motion.div
                                layoutId="activeNav"
                                className="absolute inset-0 bg-gradient-to-t lg:bg-gradient-to-r from-[#D4AF37]/20 to-transparent rounded-xl"
                                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                            />
                        )}

                        {/* Icon & Label */}
                        <item.icon size={20} className={`relative z-10 transition-transform duration-300 mb-1 lg:mb-0 ${activeTab === item.id ? 'text-[#D4AF37] scale-110' : 'group-hover:text-[#D4AF37] group-hover:scale-110'}`} />
                        <span className={`relative z-10 font-medium text-[10px] lg:text-sm tracking-wide ${activeTab === item.id ? 'text-[#D4AF37]' : ''}`}>
                            {item.label}
                        </span>

                        {/* Active Indicator Line (Desktop: Left, Mobile: Bottom) */}
                        {activeTab === item.id && (
                            <motion.div
                                layoutId="activeInd"
                                className="absolute lg:left-0 lg:top-1/2 lg:-translate-y-1/2 lg:w-1 lg:h-6 bottom-0 lg:bottom-auto w-12 h-1 lg:w-1 bg-[#D4AF37] rounded-full shadow-[0_0_10px_#D4AF37]"
                            />
                        )}
                    </button>
                ))}
            </div>

            {/* Footer (Desktop Only) */}
            <div className="mt-auto px-4 pt-6 border-t border-white/5 hidden lg:block">
                <div className="flex items-center gap-2 text-xs text-[#8B949E]/60 hover:text-[#D4AF37] transition-colors cursor-pointer">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    System Online
                </div>
            </div>
        </div>
    );
};

export default Navbar;
