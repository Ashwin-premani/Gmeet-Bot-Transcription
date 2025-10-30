import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "./ThemeToggle";
import { User, Building, Shield, LogOut } from "lucide-react";
import { logout } from "@/lib/api";

interface NavigationProps {
  userRole: 'student' | 'employer' | 'admin';
  userName: string;
  onLogout?: () => void;
}

export default function Navigation({ userRole, userName, onLogout }: NavigationProps) {
  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'student': return User;
      case 'employer': return Building;
      case 'admin': return Shield;
      default: return User;
    }
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'student': return 'Student Dashboard';
      case 'employer': return 'Employer Dashboard';
      case 'admin': return 'Admin Dashboard';
      default: return 'Dashboard';
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      onLogout?.();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const RoleIcon = getRoleIcon(userRole);

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'student': return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case 'employer': return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case 'admin': return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  return (
    <nav className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-6">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 hover-elevate rounded-md px-2 py-1">
            <div className="w-8 h-8 bg-primary rounded-md flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">AI</span>
            </div>
            <span className="font-semibold text-lg">Smart Allocation</span>
          </Link>

          {/* Current Role */}
          <div className="relative">
            <div className="flex items-center gap-2 px-3 py-2 text-sm">
              <RoleIcon className="h-4 w-4" />
              <span>{getRoleLabel(userRole)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* User Info */}
          <div className="flex items-center gap-2">
            <Badge className={getRoleColor(userRole)} data-testid="badge-user-role">
              {userRole.charAt(0).toUpperCase() + userRole.slice(1)}
            </Badge>
            <span className="text-sm font-medium">{userName}</span>
          </div>

          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Logout Button */}
          <Button variant="outline" onClick={handleLogout} data-testid="button-logout">
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </Button>
        </div>
      </div>
    </nav>
  );
}