/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,js}",
    "./node_modules/flowbite/**/*.js"
  ],
  theme: {
    extend: {
      color:{
      primary:'#CD5C08'
      }
    },
  },
  plugins: [
    require('flowbite/plugin')
  ],
}

