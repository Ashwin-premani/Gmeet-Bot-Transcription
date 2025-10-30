import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import MainLayout from "@/components/MainLayout";
import LandingPage from "@/pages/landing";
import AuthPage from "@/pages/auth";
import PostInternshipPage from "@/pages/post-internship";
import InternshipDetailsPage from "@/pages/internship-details";
import EmployerInternshipDetailsPage from "@/pages/employer-internship-details";
import StudentDetailsPage from "@/pages/student-details";
import NotFound from "@/pages/not-found";

function Router() {
  return (
    <Switch>
      {/* Landing page */}
      <Route path="/" component={LandingPage} />
      {/* Authentication */}
      <Route path="/auth" component={AuthPage} />
      {/* Dashboard routes */}
      <Route path="/dashboard" component={MainLayout} />
      <Route path="/student" component={MainLayout} />
      <Route path="/employer" component={MainLayout} />
      <Route path="/admin" component={MainLayout} />
      {/* Post internship */}
      <Route path="/post-internship" component={PostInternshipPage} />
      {/* Detail pages */}
      <Route path="/internship/:id" component={InternshipDetailsPage} />
      <Route path="/employer/internship/:id" component={EmployerInternshipDetailsPage} />
      <Route path="/student/:id" component={StudentDetailsPage} />
      {/* Fallback to 404 */}
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="light" storageKey="ai-allocation-theme">
        <TooltipProvider>
          <Router />
          <Toaster />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
