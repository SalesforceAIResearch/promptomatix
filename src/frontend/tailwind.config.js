/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./src/components/CommentPopover.jsx",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3b82f6', // blue-500
          foreground: '#ffffff',
        },
        accent: {
          DEFAULT: '#f3f4f6', // gray-100
          foreground: '#1f2937', // gray-800
        },
        'comment-highlight': 'rgba(255, 255, 0, 0.3)',
      }
    },
  },
  plugins: [],
}