import { useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import Navigation from "./Navigation";
import EmployerDashboardContainer from "./EmployerDashboardContainer";
import StudentDashboardContainer from "./StudentDashboardContainer";
import { me } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";

export default function MainLayout() {
  const [, setLocation] = useLocation();
  
  const { data: auth, isLoading } = useQuery({
    queryKey: ["auth", "me"],
    queryFn: me,
    retry: false,
  });

  const handleLogout = () => {
    setLocation("/auth");
    // clear query cache
    queryClient.clear();
    // remove session
    sessionStorage.removeItem("session");
    // remove cookies
    document.cookie.split(";").forEach(function(c) { 
      document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/");
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-lg">Loading...</div>
      </div>
    );
  }

  if (!auth?.user) {
    setLocation("/auth");
    return null;
  }

  const userRole = auth.user.role as 'student' | 'employer' | 'admin';
  const userName = auth.user.name || auth.profile?.companyName || 'User';

  const renderDashboard = () => {
    switch (userRole) {
      case 'employer':
        return <EmployerDashboardContainer />;
      case 'student':
        return <StudentDashboardContainer />;
      case 'admin':
        return (
          <div className="p-6">
            <h1 className="text-2xl font-bold">Admin Dashboard</h1>
            <p className="text-muted-foreground">Coming soon - will show metrics and data from database</p>
          </div>
        );
      default:
        return <div className="p-6">Dashboard not available</div>;
    }
  };

  return (
    <div className="min-h-screen bg-background" data-testid="layout-main">
      <Navigation
        userRole={userRole}
        userName={userName}
        onLogout={handleLogout}
      />
      <main className="min-h-[calc(100vh-4rem)]">
        {renderDashboard()}
      </main>
    </div>
  );
}