/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#fef3e2",
          100: "#fde4b9",
          200: "#fcd48c",
          300: "#fbc45f",
          400: "#fab83d",
          500: "#f9a825",  // warm saffron/gold
          600: "#f59b21",
          700: "#ef8a1b",
          800: "#e97a16",
          900: "#df5f0d",
        },
      },
    },
  },
  plugins: [],
};
