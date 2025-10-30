# AI-Based Smart Allocation Engine for PM Internship Scheme

## Overview

The AI-Based Smart Allocation Engine is a comprehensive web application designed to streamline the internship allocation process for Product Management positions. The system leverages AI-powered matching algorithms to connect students with relevant internship opportunities while ensuring fair distribution and quota compliance. The platform serves three primary user types: students seeking internships, employers posting opportunities, and administrators monitoring system performance and fairness metrics.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Framework**: React with TypeScript using Vite as the build tool and development server
- **UI Library**: Radix UI primitives with shadcn/ui components for consistent, accessible interface elements
- **Styling**: Tailwind CSS with custom design system implementing Material Design principles
- **State Management**: TanStack Query (React Query) for server state management and API caching
- **Routing**: Wouter for lightweight client-side routing
- **Form Handling**: React Hook Form with Zod validation for type-safe form management
- **Theme System**: Custom theme provider supporting light/dark mode with CSS custom properties

### Backend Architecture
**Runtime**: Node.js with Express.js server framework
- **Language**: TypeScript for type safety across the entire codebase
- **Session Management**: Express-session with PostgreSQL session store for user authentication
- **Password Security**: bcrypt for secure password hashing
- **API Design**: RESTful API structure with type-safe request/response handling

### Database Layer
**ORM**: Drizzle ORM for type-safe database operations
- **Database**: PostgreSQL (configured for Neon serverless)
- **Schema Management**: Centralized schema definitions with automated migrations
- **Connection Pooling**: Neon serverless connection pooling for scalable database access

### Core Data Models
- **Users**: Base authentication and profile information with role-based access (student/employer/admin)
- **Students**: Academic profiles including university, major, skills, and preferences
- **Employers**: Company information and internship posting capabilities
- **Internships**: Job postings with requirements and application tracking
- **Applications**: Student-internship relationship tracking with status management
- **AI Matches**: Machine learning-driven compatibility scoring and recommendation engine

### Authentication & Authorization
**Session-based Authentication**: Server-side session management with PostgreSQL persistence
- **Role-based Access Control**: Three distinct user roles with appropriate permission levels
- **Security**: HTTP-only cookies with CSRF protection and secure session configuration

### AI Matching System
**Smart Allocation Engine**: Core matching algorithm that analyzes student profiles against internship requirements
- **Compatibility Scoring**: Numerical matching scores (0-100) based on skills, location, and preferences
- **Fairness Monitoring**: Built-in quota compliance and bias detection systems
- **Recommendation Engine**: Personalized internship suggestions for students and candidate recommendations for employers

### Design System
**Component Architecture**: Modular component library based on shadcn/ui with custom extensions
- **Theme Implementation**: CSS custom properties enabling consistent theming across light/dark modes
- **Typography**: Inter font family for body text, JetBrains Mono for data display
- **Color System**: Professional blue primary palette with semantic color assignments
- **Responsive Design**: Mobile-first approach with Tailwind breakpoints

## External Dependencies

### Database Services
- **Neon Database**: Serverless PostgreSQL hosting with connection pooling and automated scaling
- **Drizzle Kit**: Database migration and schema management tooling

### UI Component Libraries
- **Radix UI**: Headless, accessible component primitives for complex UI elements
- **Recharts**: Data visualization library for admin analytics and quota compliance charts
- **Lucide React**: Consistent icon library with comprehensive symbol coverage

### Development Tools
- **Vite**: Fast development server and build tool with hot module replacement
- **TypeScript**: Static type checking across frontend, backend, and shared code
- **Tailwind CSS**: Utility-first CSS framework for rapid UI development
- **ESBuild**: Fast JavaScript bundling for production builds

### Authentication & Security
- **bcrypt**: Industry-standard password hashing for secure credential storage
- **express-session**: Server-side session management with configurable stores
- **connect-pg-simple**: PostgreSQL session store adapter for persistent sessions

### Form & Data Handling
- **React Hook Form**: Performant form library with minimal re-renders
- **Zod**: TypeScript-first schema validation for forms and API endpoints
- **TanStack Query**: Server state management with caching, synchronization, and background updates

### Fonts & Assets
- **Google Fonts**: Inter and JetBrains Mono font families loaded via CDN
- **Font Optimization**: Preconnect headers and display=swap for improved loading performance