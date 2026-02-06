import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, Info } from 'lucide-react';

const Toast = ({ message, type = 'success', onClose }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onClose();
        }, 3000);
        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-8 right-8 z-[120] flex items-center gap-3 px-6 py-4 rounded-xl glass-panel border-[#D4AF37]/20 shadow-2xl"
        >
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${type === 'success' ? 'bg-[#D4AF37] text-black' : 'bg-white/10 text-white'}`}>
                {type === 'success' ? <Check size={16} /> : <Info size={16} />}
            </div>
            <div>
                <p className="font-playfair font-bold text-white text-sm">{type === 'success' ? 'Success' : 'Note'}</p>
                <p className="text-[#8B949E] text-xs">{message}</p>
            </div>
        </motion.div>
    );
};

export default Toast;
