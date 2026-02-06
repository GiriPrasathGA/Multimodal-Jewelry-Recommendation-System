import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const AuraCursor = () => {
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isPointer, setIsPointer] = useState(false);
    const [isClicking, setIsClicking] = useState(false);

    useEffect(() => {
        const handleMouseMove = (e) => {
            setPosition({ x: e.clientX, y: e.clientY });

            const target = e.target;
            // Check if hovering over clickable elements
            const clickable =
                target.tagName === 'BUTTON' ||
                target.tagName === 'A' ||
                target.closest('button') ||
                target.closest('a') ||
                window.getComputedStyle(target).cursor === 'pointer';

            setIsPointer(clickable);
        };

        const handleMouseDown = () => setIsClicking(true);
        const handleMouseUp = () => setIsClicking(false);

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    return (
        <>
            {/* Precision Diamond Core */}
            <motion.div
                className="fixed top-0 left-0 w-3 h-3 bg-[#D4AF37] pointer-events-none z-[9999] shadow-[0_0_10px_#D4AF37]"
                style={{ willChange: 'transform' }}
                animate={{
                    x: position.x - 6,
                    y: position.y - 6,
                    rotate: 45, // Diamond shape
                    scale: isClicking ? 0.8 : 1
                }}
                transition={{ type: 'tween', duration: 0 }}
            />

            {/* Click Ripple Burst */}
            <motion.div
                className="fixed top-0 left-0 w-10 h-10 border-2 border-[#D4AF37] rounded-full pointer-events-none z-[9997]"
                style={{ willChange: 'transform, opacity' }}
                animate={{
                    x: position.x - 20,
                    y: position.y - 20,
                    scale: isClicking ? 2.5 : 1,
                    opacity: isClicking ? 0 : 1,
                    borderColor: isClicking ? '#FFFFFF' : '#D4AF37'
                }}
                transition={{ duration: 0.4 }}
            />

            {/* Outer Targeting Reticle (Only visible on clickable) */}
            <motion.div
                className="fixed top-0 left-0 w-10 h-10 border border-[#D4AF37] pointer-events-none z-[9998]"
                style={{ willChange: 'transform, opacity' }}
                animate={{
                    x: position.x - 20,
                    y: position.y - 20,
                    rotate: isPointer ? 45 : 0,
                    scale: isPointer ? 1.2 : 0.5,
                    opacity: isPointer ? 1 : 0,
                    borderRadius: isPointer ? '0%' : '50%' // Morph from circle to square
                }}
                transition={{ type: 'spring', stiffness: 200, damping: 20 }}
            />

            {/* Trailing Glow Aura */}
            <motion.div
                className="fixed top-0 left-0 w-64 h-64 bg-[#D4AF37] rounded-full pointer-events-none z-[0] mix-blend-screen filter blur-[80px]"
                style={{ willChange: 'transform' }}
                animate={{
                    x: position.x - 128,
                    y: position.y - 128,
                    opacity: 0.15,
                    scale: isClicking ? 1.5 : 1
                }}
                transition={{ type: 'spring', stiffness: 50, damping: 50, mass: 1 }}
            />
        </>
    );
};

export default AuraCursor;
