import Navigation from '../Navigation';

export default function NavigationExample() {
  return (
    <div className="bg-background">
      <Navigation
        userRole="student"
        userName="Sarah Chen"
        onRoleSwitch={(role) => console.log('Switch to role:', role)}
      />
      <div className="p-8">
        <h2 className="text-xl font-semibold mb-4">Navigation Component</h2>
        <p className="text-muted-foreground">
          This navigation bar allows users to switch between different dashboard views (Student, Employer, Admin)
          and includes theme toggle and user profile access.
        </p>
      </div>
    </div>
  );
}