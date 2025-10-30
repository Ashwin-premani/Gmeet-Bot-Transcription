# Design Guidelines: AI-Based Smart Allocation Engine for PM Internship Scheme

## Design Approach
**System-Based Approach** using Material Design principles adapted for a professional productivity tool. This utility-focused application prioritizes efficiency, data clarity, and user learnability over visual aesthetics.

## Core Design Elements

### A. Color Palette
**Light Mode:**
- Primary: 220 91% 35% (Professional blue)
- Secondary: 210 12% 16% (Dark slate)
- Success: 142 76% 36% (Green for matches)
- Warning: 38 92% 50% (Amber for pending)
- Error: 0 84% 60% (Red for rejections)
- Background: 0 0% 98% (Near white)
- Surface: 0 0% 100% (Pure white)

**Dark Mode:**
- Primary: 220 91% 65% (Lighter blue)
- Secondary: 210 12% 84% (Light slate)
- Success: 142 76% 56% (Brighter green)
- Warning: 38 92% 60% (Brighter amber)
- Error: 0 84% 70% (Lighter red)
- Background: 210 12% 8% (Dark slate)
- Surface: 210 12% 12% (Slightly lighter dark)

### B. Typography
- **Primary Font:** Inter (Google Fonts)
- **Headers:** Font weights 600-700, sizes 2xl-4xl
- **Body Text:** Font weight 400, size base-lg
- **Captions:** Font weight 500, size sm-xs
- **Monospace:** JetBrains Mono for fit scores and data

### C. Layout System
**Spacing Units:** Tailwind units 2, 4, 6, 8, 12, 16
- Tight spacing (p-2, m-2) for form elements
- Medium spacing (p-4, gap-4) for card content
- Large spacing (p-8, mb-12) for section separation

### D. Component Library

**Navigation:**
- Horizontal tab-based navigation for role switching
- Sidebar navigation within each dashboard
- Breadcrumb trails for deep navigation

**Match Cards:**
- Clean white/dark surface cards with subtle shadows
- Circular progress indicators for fit scores (0-100%)
- Color-coded status badges (green/amber/red)
- "Why this match?" tooltip triggers

**Data Displays:**
- Minimalist table designs with alternating row colors
- Interactive donut charts for quota compliance
- Linear progress bars for capacity utilization
- Status indicators using color-coded dots

**Forms & Actions:**
- Outlined input fields with floating labels
- Primary action buttons (filled)
- Secondary action buttons (outlined)
- Bulk action checkboxes with batch operation bar

**Charts & Visualizations:**
- Clean line charts for trends
- Donut charts for quota breakdowns
- Bar charts for capacity metrics
- Consistent color coding across all charts

### E. Animations
**Minimal Motion:**
- Subtle hover states on interactive elements
- Smooth transitions for tab switching (200ms)
- Loading spinners for data fetching
- No decorative animations or excessive motion

## Dashboard-Specific Guidelines

**Student Dashboard:**
- Grid layout for internship match cards
- Prominent fit scores with visual progress rings
- Application status timeline
- Quick actions panel

**Employer Dashboard:**
- Split layout: posted internships left, candidates right
- Candidate recommendation cards with ranking
- Bulk selection interface for candidate actions
- Internship performance metrics

**Admin Dashboard:**
- Multi-column layout for KPIs
- Interactive charts prominently displayed
- Real-time status indicators
- Compliance monitoring alerts

## Responsive Behavior
- Mobile: Single column layouts, collapsible navigation
- Tablet: Two-column layouts where appropriate
- Desktop: Full multi-column layouts with sidebars

## Accessibility
- WCAG 2.1 AA compliance
- High contrast ratios in both light and dark modes
- Keyboard navigation support
- Screen reader compatibility
- Focus indicators on all interactive elements

## Images
No hero images or decorative imagery. Focus on:
- Simple avatar placeholders for user profiles
- Company logos (small, consistent sizing)
- Icon library: Heroicons for consistency
- Data visualization charts and graphs as primary visual elements

The design emphasizes data clarity, professional appearance, and efficient user workflows over visual flair.