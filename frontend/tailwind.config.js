/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'jewel-dark': '#0B0D12',
        'jewel-card': '#161B22',
        'jewel-gold': '#D4AF37',
        'jewel-gold-light': '#F3E5AB',
      },
      fontFamily: {
        // Keeping key names to match existing class usage, but swapping the font family
        playfair: ['"Bodoni Moda"', 'serif'],
        inter: ['"Montserrat"', 'sans-serif'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gold-shiny': 'linear-gradient(45deg, #FFD700, #FDB931, #FFFFFF, #9E7922, #FFD700)',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        'float': 'float 6s ease-in-out infinite',
        'gold-shimmer': 'shimmer 3s linear infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '200% center' },
          '100%': { backgroundPosition: '-200% center' },
        }
      },
    },
  },
  plugins: [],
}
