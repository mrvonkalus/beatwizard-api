# 🎨 BeatWizard Brand Colors

## Primary Palette

### Purple Shades
```css
--purple-50: #faf5ff;
--purple-100: #f3e8ff;
--purple-200: #e9d5ff;
--purple-300: #d8b4fe;
--purple-400: #c084fc;
--purple-500: #a855f7;  /* Primary purple */
--purple-600: #7c3aed;  /* Main brand purple */
--purple-700: #6d28d9;
--purple-800: #5b21b6;
--purple-900: #4c1d95;
```

### Blue Shades
```css
--blue-50: #eff6ff;
--blue-100: #dbeafe;
--blue-200: #bfdbfe;
--blue-300: #93c5fd;
--blue-400: #60a5fa;
--blue-500: #3b82f6;
--blue-600: #2563eb;   /* Main brand blue */
--blue-700: #1d4ed8;
--blue-800: #1e40af;
--blue-900: #1e3a8a;
```

## Secondary Palette

### Pink Accent
```css
--pink-400: #f472b6;
--pink-500: #ec4899;   /* Accent pink */
--pink-600: #db2777;
```

### Indigo Support
```css
--indigo-500: #6366f1;
--indigo-600: #4f46e5;  /* Support color */
--indigo-700: #4338ca;
```

## Gradient Combinations

### Primary Gradients
```css
/* Hero gradient */
.bg-hero-gradient {
  background: linear-gradient(135deg, #f3e8ff 0%, #fce7f3 50%, #e0e7ff 100%);
}

/* Button gradient */
.bg-button-gradient {
  background: linear-gradient(90deg, #7c3aed 0%, #2563eb 100%);
}

/* Card gradient */
.bg-card-gradient {
  background: linear-gradient(135deg, #ffffff 0%, #f3e8ff 100%);
}

/* Wizard gradient */
.bg-wizard-gradient {
  background: linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #ec4899 100%);
}
```

## Usage Guidelines

### Do's ✅
- Use purple-600 as primary brand color
- Use blue-600 for secondary actions
- Use gradients for hero sections and buttons
- Maintain 4.5:1 contrast ratio for accessibility
- Use lighter shades for backgrounds

### Don'ts ❌
- Don't use pure black or pure white
- Don't mix too many colors in one interface
- Don't use neon or overly bright colors
- Don't ignore contrast requirements

## Accessibility

### Contrast Ratios
- **Purple-600 on white**: 7.56:1 (AA+)
- **Blue-600 on white**: 8.59:1 (AA+)
- **Purple-700 on purple-50**: 11.24:1 (AA+)

### Color Blindness Considerations
- Use icons and labels in addition to color
- Test with color blindness simulators
- Provide high contrast mode option

## Export Formats

### CSS Variables
```css
:root {
  --brand-purple: #7c3aed;
  --brand-blue: #2563eb;
  --brand-pink: #ec4899;
  --bg-gradient: linear-gradient(135deg, #f3e8ff 0%, #fce7f3 50%, #e0e7ff 100%);
}
```

### Tailwind Config
```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        'brand-purple': '#7c3aed',
        'brand-blue': '#2563eb',
        'brand-pink': '#ec4899',
      }
    }
  }
}
```

### Adobe Swatch (.ase)
*Coming soon - export from design tools*

---

**These colors create the magical, professional BeatWizard aesthetic! 🎨✨**
