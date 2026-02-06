import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const LiveRatesTicker = () => {
    // Initial baselines for simulation (INR)
    const [rates, setRates] = useState({
        gold: { price: 62450.00, change: 125.50 },    // per 10g
        silver: { price: 74500.00, change: -240.00 },  // per kg
        diamond: { price: 425000.00, change: 1500.00 } // 1ct VVS
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setRates(prev => {
                const fluctuate = (val, volatility) => val + (Math.random() - 0.5) * volatility;

                return {
                    gold: {
                        price: fluctuate(prev.gold.price, 50.0),
                        change: fluctuate(prev.gold.change, 10.0)
                    },
                    silver: {
                        price: fluctuate(prev.silver.price, 100.0),
                        change: fluctuate(prev.silver.change, 20.0)
                    },
                    diamond: {
                        price: fluctuate(prev.diamond.price, 500.0),
                        change: fluctuate(prev.diamond.change, 50.0)
                    }
                };
            });
        }, 3000); // Update every 3 seconds

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex gap-4 mt-6 lg:mt-12 overflow-x-auto pb-4 -mx-6 px-6 lg:mx-0 lg:px-0 lg:pb-0 no-scrollbar snap-x">
            <RateCard
                label="Gold (10g)"
                code="MCX"
                data={rates.gold}
                color="text-[#D4AF37]"
            />
            <RateCard
                label="Silver (1kg)"
                code="MCX"
                data={rates.silver}
                color="text-gray-300"
            />
            <RateCard
                label="Diamond (1ct)"
                code="VVS1"
                data={rates.diamond}
                color="text-blue-200"
            />
        </div>
    );
};

const RateCard = ({ label, code, data, color }) => (
    <div className="flex flex-col gap-2 px-6 py-5 rounded-xl glass-panel border border-white/5 min-w-[240px] transition-all duration-500 hover:border-[#D4AF37]/40 hover:bg-[#D4AF37]/5 hover:shadow-[0_0_30px_-10px_rgba(212,175,55,0.15)] group relative overflow-hidden">
        <div className="flex justify-between items-start relative z-10">
            <p className="text-[#8B949E] text-[9px] font-bold tracking-[0.25em] uppercase mb-1 font-inter">{label}</p>
            <div className="flex items-center gap-2">
                <span className="w-1 h-3 bg-white/10 rounded-full group-hover:bg-[#D4AF37] transition-colors" />
                <span className="text-[9px] text-white/40 font-inter tracking-widest uppercase">{code}</span>
            </div>
        </div>

        <div className="flex items-baseline gap-3 mt-1 relative z-10">
            <span className={`text-4xl font-playfair italic font-medium ${color} drop-shadow-md`}>
                ₹{data.price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </span>
        </div>

        <div className={`flex items-center text-[10px] font-bold tracking-wider mt-2 relative z-10 ${data.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {data.change >= 0 ? <TrendingUp size={14} className="mr-1.5" /> : <TrendingDown size={14} className="mr-1.5" />}
            ₹{Math.abs(data.change).toFixed(2)}
            <span className="text-white/20 ml-auto text-[8px] font-normal uppercase tracking-widest border border-white/10 px-2 py-0.5 rounded-full">Live Update</span>
        </div>

        {/* Background Gradient Pulse */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/0 via-white/0 to-white/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
    </div>
);

export default LiveRatesTicker;
