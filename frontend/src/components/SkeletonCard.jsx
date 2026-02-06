import React from 'react';

const SkeletonCard = () => {
    return (
        <div className="relative h-[420px] rounded-2xl overflow-hidden bg-[#161B22] border border-white/5">
            {/* Shimmer Effect - Gold Dust */}
            <div className="absolute inset-0 z-10 animate-gold-shimmer bg-gradient-to-r from-transparent via-[#D4AF37]/10 to-transparent"
                style={{ backgroundSize: '200% 100%' }} />

            {/* Image Placeholder */}
            <div className="h-3/4 bg-white/5" />

            {/* Content Placeholder */}
            <div className="p-6 space-y-4">
                <div className="h-6 w-3/4 bg-white/5 rounded mx-auto" />
                <div className="h-4 w-1/2 bg-white/5 rounded mx-auto" />
            </div>
        </div>
    );
};

export default SkeletonCard;
