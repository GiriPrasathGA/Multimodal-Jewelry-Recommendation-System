import React, { useRef, useState, useMemo } from 'react';
import { motion, useMotionValue, useSpring, useTransform, useMotionTemplate } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';

const ProductCard = ({ item, onClick }) => {
    const ref = useRef(null);
    const [hover, setHover] = useState(false);

    // 1. Motion Values for Sheen (Pixel coordinates)
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    // 2. Motion Values for 3D Tilt (Percentage -0.5 to 0.5)
    const x = useMotionValue(0);
    const y = useMotionValue(0);

    const mouseXSpring = useSpring(x);
    const mouseYSpring = useSpring(y);

    const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["7deg", "-7deg"]);
    const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-7deg", "7deg"]);

    // Memoize random price to keep it stable across renders
    const price = useMemo(() => {
        // Generate price between 2.5 Lakhs and 12 Lakhs
        const randomPrice = Math.floor(Math.random() * (1200000 - 250000 + 1)) + 250000;
        return randomPrice.toLocaleString('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0
        });
    }, [item.id]);

    // Create sheen gradient (Moved to top level to respect Hook rules)
    const sheenGradient = useMotionTemplate`
        radial-gradient(
            650px circle at ${mouseX}px ${mouseY}px,
            rgba(255,255,255,0.1),
            transparent 40%
        ),
        radial-gradient(
            400px circle at ${mouseX}px ${mouseY}px,
            rgba(212,175,55,0.15),
            transparent 30%
        )
    `;

    const handleMouseMove = (e) => {
        if (!ref.current) return;

        const rect = ref.current.getBoundingClientRect();

        const width = rect.width;
        const height = rect.height;

        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;

        // Update raw pixel values for gradient
        mouseX.set(clientX);
        mouseY.set(clientY);

        const xPct = clientX / width - 0.5;
        const yPct = clientY / height - 0.5;

        x.set(xPct);
        y.set(yPct);
    };

    const handleMouseLeave = () => {
        setHover(false);
        x.set(0);
        y.set(0);
    };

    return (
        <motion.div
            ref={ref}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            onMouseEnter={() => setHover(true)}
            style={{
                rotateX,
                rotateY,
                transformStyle: "preserve-3d",
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            onClick={() => onClick(item)}
            className="group relative h-[450px] rounded-2xl cursor-pointer perspective-1000"
        >
            <div
                style={{ transform: "translateZ(75px)", transformStyle: "preserve-3d" }}
                className="absolute inset-0 rounded-2xl overflow-hidden shadow-black/40 shadow-xl border border-white/5 transition-all duration-300 group-hover:shadow-[0_20px_60px_-15px_rgba(212,175,55,0.3)] bg-[#0B0D12]"
            >
                {/* Background Image - Slight zoom on hover */}
                <div className="absolute inset-0 bg-black">
                    <img
                        src={`data:image/jpeg;base64,${item.image_base64}`}
                        alt={item.category}
                        className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 opacity-90"
                    />
                </div>

                {/* Dynamic Holographic Sheen Overlay */}
                <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-20"
                    style={{
                        background: sheenGradient,
                        mixBlendMode: 'overlay',
                    }}
                />

                {/* Overlay Gradient */}
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent opacity-80 transition-opacity duration-300 group-hover:opacity-95 z-10" />

                {/* Match Badge */}
                <div className="absolute top-4 right-4 bg-black/40 backdrop-blur-md px-3 py-1 rounded-full border border-[#D4AF37]/30 text-[#D4AF37] text-xs font-bold tracking-wider shadow-lg z-30 transform translate-z-20">
                    {Math.round(item.score * 100)}% MATCH
                </div>

                {/* Content Slide-up */}
                <div
                    style={{ transform: "translateZ(50px)" }}
                    className="absolute bottom-0 left-0 right-0 p-6 z-30"
                >
                    <div className="flex flex-col gap-1 mb-3">
                        <span className="text-3xl font-playfair italic text-[#D4AF37] font-medium">{price}</span>
                        <h3 className="text-white/90 text-[10px] font-bold uppercase tracking-[0.25em] leading-relaxed">
                            {item.category || "Fine Jewelry"}
                        </h3>
                    </div>

                    {/* Animated Divider */}
                    <div className="h-0.5 w-12 bg-[#D4AF37] mb-3 transition-all duration-300 group-hover:w-full opacity-60" />

                    <p className="text-[#E0E0E0]/80 text-sm font-light leading-relaxed line-clamp-2 mb-4">
                        {item.description}
                    </p>

                    <button className="w-full py-3 bg-white text-black font-bold uppercase tracking-widest text-xs rounded-lg transform translate-y-8 opacity-0 group-hover:translate-y-0 group-hover:opacity-100 transition-all duration-300 flex items-center justify-center gap-2 hover:bg-[#D4AF37]">
                        Inquire Concierge
                        <ArrowUpRight size={14} />
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

export default ProductCard;
