import React from 'react';
import ProductCard from './ProductCard';
import SkeletonCard from './SkeletonCard';
import { motion } from 'framer-motion';

const container = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: {
            staggerChildren: 0.1
        }
    }
};

const itemAnim = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0 }
};

const ResultsGrid = ({ results, onCardClick, loading, onSimilar }) => {
    if (loading) {
        return (
            <div className='mt-8'>
                <div className='flex items-center gap-4 mb-8 opacity-50'>
                    <div className='h-px bg-white/20 flex-grow'></div>
                    <span className='text-jewel-gold text-xs tracking-widest uppercase animate-pulse'>Curating Selection...</span>
                    <div className='h-px bg-white/20 flex-grow'></div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {[...Array(6)].map((_, i) => <SkeletonCard key={i} />)}
                </div>
            </div>
        );
    }

    if (!results || results.length === 0) return null;

    return (
        <div className='mt-8'>
            <div className='flex items-center gap-4 mb-8 animate-fade-in-up'>
                <div className='h-px bg-gradient-to-r from-transparent via-[#D4AF37] to-transparent flex-grow opacity-50'></div>
                <span className='text-gold-gradient font-playfair font-bold tracking-widest uppercase text-sm'>Curated Recommendations</span>
                <div className='h-px bg-gradient-to-r from-transparent via-[#D4AF37] to-transparent flex-grow opacity-50'></div>
            </div>

            <motion.div
                variants={container}
                initial="hidden"
                animate="show"
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
            >
                {[...results].sort((a, b) => (b.score || 0) - (a.score || 0)).map((item, idx) => (
                    <motion.div key={idx} variants={itemAnim}>
                        <ProductCard item={item} onClick={onCardClick} onSimilar={onSimilar} />
                    </motion.div>
                ))}
            </motion.div>
        </div>
    );
};

export default ResultsGrid;
