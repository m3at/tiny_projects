/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.{html,js}"],
  theme: {
    extend: {
        boxShadow: {
            // Overwrite some defaults
            // Infos:
            // https://tailwindcss.com/docs/customizing-colors
            // https://tailwindcss.com/docs/box-shadow
            // https://tailwindcss.com/docs/box-shadow-color
            'DEFAULT': '0 1px 3px 0 #27272a, 0 1px 2px -1px #27272a;',
            'md': '0 4px 6px -1px rgb(187 70 10 / 0.5), 0 2px 4px -2px rgb(187 70 10 / 0.5);',
        }
    },
  },
  plugins: [],
}

