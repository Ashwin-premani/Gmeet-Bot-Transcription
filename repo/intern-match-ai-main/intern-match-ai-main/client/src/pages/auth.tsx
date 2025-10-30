import { useState } from "react";
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ThemeToggle } from "@/components/ThemeToggle";
import { User, Building, ArrowLeft, Eye, EyeOff } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { login as apiLogin, registerStudent, registerEmployer } from "@/lib/api";

const loginSchema = z.object({
  email: z.string().email("Please enter a valid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  userType: z.enum(["student", "employer"])
});

const studentSignupSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Please enter a valid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  confirmPassword: z.string(),
  university: z.string().min(2, "University is required"),
  major: z.string().min(2, "Major is required"),
  graduationYear: z.number().min(2024).max(2030),
  gpa: z.string().optional(),
  skills: z.string().min(10, "Please describe your skills (at least 10 characters)"),
  interests: z.string().min(10, "Please describe your interests (at least 10 characters)")
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

const employerSignupSchema = z.object({
  name: z.string().min(2, "Contact name is required"),
  email: z.string().email("Please enter a valid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
  confirmPassword: z.string(),
  companyName: z.string().min(2, "Company name is required"),
  industry: z.string().min(2, "Industry is required"),
  companySize: z.string(),
  location: z.string().min(2, "Location is required"),
  website: z.string().url("Please enter a valid website URL").optional().or(z.literal("")),
  description: z.string().min(20, "Please provide a brief company description (at least 20 characters)")
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export default function AuthPage() {
  const [, setLocation] = useLocation();
  const [isLogin, setIsLogin] = useState(true);
  const [userType, setUserType] = useState<"student" | "employer">("student");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const loginForm = useForm({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
      userType: "student" as const
    }
  });

  const studentForm = useForm({
    resolver: zodResolver(studentSignupSchema),
    defaultValues: {
      name: "",
      email: "",
      password: "",
      confirmPassword: "",
      university: "",
      major: "",
      graduationYear: 2025,
      gpa: "",
      skills: "",
      interests: ""
    }
  });

  const employerForm = useForm({
    resolver: zodResolver(employerSignupSchema),
    defaultValues: {
      name: "",
      email: "",
      password: "",
      confirmPassword: "",
      companyName: "",
      industry: "",
      companySize: "",
      location: "",
      website: "",
      description: ""
    }
  });

  const onLogin = async (values: z.infer<typeof loginSchema>) => {
    try {
      const { user } = await apiLogin(values);
      const path = user.role === "student" ? "/student" : user.role === "employer" ? "/employer" : "/admin";
      setLocation(path);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Login failed";
      loginForm.setError("email", { message });
    }
  };

  const onStudentSignup = async (values: z.infer<typeof studentSignupSchema>) => {
    try {
      const { confirmPassword, ...payload } = values;
      const { user } = await registerStudent(payload);
      const path = "/student";
      setLocation(path);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Registration failed";
      studentForm.setError("email", { message });
    }
  };

  const onEmployerSignup = async (values: z.infer<typeof employerSignupSchema>) => {
    try {
      const { confirmPassword, ...payload } = values;
      const { user } = await registerEmployer(payload);
      const path = "/employer";
      setLocation(path);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Registration failed";
      employerForm.setError("email", { message });
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col" data-testid="page-auth">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur">
        <div className="container mx-auto flex h-16 items-center justify-between px-6">
          <Link href="/" className="flex items-center gap-2 hover-elevate rounded-md px-2 py-1">
            <ArrowLeft className="h-4 w-4" />
            <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-xs">AI</span>
            </div>
            <span className="font-semibold">Back to Home</span>
          </Link>
          <ThemeToggle />
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <div className="text-center mb-8">
            <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mx-auto mb-4">
              <span className="text-primary-foreground font-bold">AI</span>
            </div>
            <h1 className="text-2xl font-bold">
              {isLogin ? "Welcome Back" : "Join Smart Allocation"}
            </h1>
            <p className="text-muted-foreground">
              {isLogin 
                ? "Sign in to access your dashboard" 
                : "Create your account to get started"
              }
            </p>
          </div>

          <Card>
            <CardHeader className="space-y-1">
              <div className="flex space-x-2">
                <Button
                  variant={isLogin ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setIsLogin(true)}
                  data-testid="button-switch-login"
                >
                  Sign In
                </Button>
                <Button
                  variant={!isLogin ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setIsLogin(false)}
                  data-testid="button-switch-signup"
                >
                  Sign Up
                </Button>
              </div>
            </CardHeader>

            <CardContent>
              {isLogin ? (
                /* Login Form */
                <Form {...loginForm}>
                  <form onSubmit={loginForm.handleSubmit(onLogin)} className="space-y-4">
                    <FormField
                      control={loginForm.control}
                      name="userType"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>I am a</FormLabel>
                          <Select onValueChange={field.onChange} defaultValue={field.value}>
                            <FormControl>
                              <SelectTrigger data-testid="select-user-type-login">
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="student">
                                <div className="flex items-center gap-2">
                                  <User className="h-4 w-4" />
                                  Student
                                </div>
                              </SelectItem>
                              <SelectItem value="employer">
                                <div className="flex items-center gap-2">
                                  <Building className="h-4 w-4" />
                                  Employer
                                </div>
                              </SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={loginForm.control}
                      name="email"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Email</FormLabel>
                          <FormControl>
                            <Input 
                              type="email" 
                              placeholder="your.email@example.com"
                              data-testid="input-email-login"
                              {...field} 
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={loginForm.control}
                      name="password"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Password</FormLabel>
                          <FormControl>
                            <div className="relative">
                              <Input 
                                type={showPassword ? "text" : "password"}
                                placeholder="Enter your password"
                                data-testid="input-password-login"
                                {...field} 
                              />
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                                onClick={() => setShowPassword(!showPassword)}
                                data-testid="button-toggle-password-login"
                              >
                                {showPassword ? (
                                  <EyeOff className="h-4 w-4" />
                                ) : (
                                  <Eye className="h-4 w-4" />
                                )}
                              </Button>
                            </div>
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <Button type="submit" className="w-full" data-testid="button-submit-login">
                      Sign In
                    </Button>
                  </form>
                </Form>
              ) : (
                /* Signup Form with Tabs */
                <Tabs value={userType} onValueChange={(value) => setUserType(value as "student" | "employer")}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="student" data-testid="tab-student">
                      <User className="h-4 w-4 mr-2" />
                      Student
                    </TabsTrigger>
                    <TabsTrigger value="employer" data-testid="tab-employer">
                      <Building className="h-4 w-4 mr-2" />
                      Employer
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="student" className="space-y-4 mt-6">
                    <Form {...studentForm}>
                      <form onSubmit={studentForm.handleSubmit(onStudentSignup)} className="space-y-4">
                        <div className="grid grid-cols-1 gap-4">
                          <FormField
                            control={studentForm.control}
                            name="name"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Full Name</FormLabel>
                                <FormControl>
                                  <Input placeholder="John Doe" data-testid="input-name-student" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <FormField
                            control={studentForm.control}
                            name="email"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Email</FormLabel>
                                <FormControl>
                                  <Input 
                                    type="email" 
                                    placeholder="john.doe@university.edu"
                                    data-testid="input-email-student"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <div className="grid grid-cols-2 gap-2">
                            <FormField
                              control={studentForm.control}
                              name="password"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Password</FormLabel>
                                  <FormControl>
                                    <div className="relative">
                                      <Input 
                                        type={showPassword ? "text" : "password"}
                                        placeholder="Password"
                                        data-testid="input-password-student"
                                        {...field} 
                                      />
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        className="absolute right-0 top-0 h-full px-2 hover:bg-transparent"
                                        onClick={() => setShowPassword(!showPassword)}
                                        data-testid="button-toggle-password-student"
                                      >
                                        {showPassword ? (
                                          <EyeOff className="h-3 w-3" />
                                        ) : (
                                          <Eye className="h-3 w-3" />
                                        )}
                                      </Button>
                                    </div>
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <FormField
                              control={studentForm.control}
                              name="confirmPassword"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Confirm</FormLabel>
                                  <FormControl>
                                    <div className="relative">
                                      <Input 
                                        type={showConfirmPassword ? "text" : "password"}
                                        placeholder="Confirm"
                                        data-testid="input-confirm-password-student"
                                        {...field} 
                                      />
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        className="absolute right-0 top-0 h-full px-2 hover:bg-transparent"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                        data-testid="button-toggle-confirm-password-student"
                                      >
                                        {showConfirmPassword ? (
                                          <EyeOff className="h-3 w-3" />
                                        ) : (
                                          <Eye className="h-3 w-3" />
                                        )}
                                      </Button>
                                    </div>
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                          </div>

                          <FormField
                            control={studentForm.control}
                            name="university"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>University</FormLabel>
                                <FormControl>
                                  <Input placeholder="Stanford University" data-testid="input-university" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <div className="grid grid-cols-2 gap-2">
                            <FormField
                              control={studentForm.control}
                              name="major"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Major</FormLabel>
                                  <FormControl>
                                    <Input placeholder="Computer Science" data-testid="input-major" {...field} />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <FormField
                              control={studentForm.control}
                              name="graduationYear"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Graduation Year</FormLabel>
                                  <FormControl>
                                    <Input 
                                      type="number" 
                                      placeholder="2025"
                                      min="2024"
                                      max="2030"
                                      data-testid="input-graduation-year"
                                      {...field}
                                      onChange={(e) => field.onChange(parseInt(e.target.value))}
                                    />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                          </div>

                          <FormField
                            control={studentForm.control}
                            name="skills"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Skills & Experience</FormLabel>
                                <FormControl>
                                  <Textarea 
                                    placeholder="Describe your technical skills, programming languages, tools, and relevant experience..."
                                    data-testid="textarea-skills"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <FormField
                            control={studentForm.control}
                            name="interests"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Career Interests</FormLabel>
                                <FormControl>
                                  <Textarea 
                                    placeholder="What areas of product management interest you? What type of companies or products would you like to work on?"
                                    data-testid="textarea-interests"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
                        </div>

                        <Button type="submit" className="w-full" data-testid="button-submit-student">
                          Create Student Account
                        </Button>
                      </form>
                    </Form>
                  </TabsContent>

                  <TabsContent value="employer" className="space-y-4 mt-6">
                    <Form {...employerForm}>
                      <form onSubmit={employerForm.handleSubmit(onEmployerSignup)} className="space-y-4">
                        <div className="grid grid-cols-1 gap-4">
                          <FormField
                            control={employerForm.control}
                            name="name"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Contact Name</FormLabel>
                                <FormControl>
                                  <Input placeholder="Jane Smith" data-testid="input-name-employer" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <FormField
                            control={employerForm.control}
                            name="email"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Business Email</FormLabel>
                                <FormControl>
                                  <Input 
                                    type="email" 
                                    placeholder="jane.smith@company.com"
                                    data-testid="input-email-employer"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <div className="grid grid-cols-2 gap-2">
                            <FormField
                              control={employerForm.control}
                              name="password"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Password</FormLabel>
                                  <FormControl>
                                    <div className="relative">
                                      <Input 
                                        type={showPassword ? "text" : "password"}
                                        placeholder="Password"
                                        data-testid="input-password-employer"
                                        {...field} 
                                      />
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        className="absolute right-0 top-0 h-full px-2 hover:bg-transparent"
                                        onClick={() => setShowPassword(!showPassword)}
                                        data-testid="button-toggle-password-employer"
                                      >
                                        {showPassword ? (
                                          <EyeOff className="h-3 w-3" />
                                        ) : (
                                          <Eye className="h-3 w-3" />
                                        )}
                                      </Button>
                                    </div>
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <FormField
                              control={employerForm.control}
                              name="confirmPassword"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Confirm</FormLabel>
                                  <FormControl>
                                    <div className="relative">
                                      <Input 
                                        type={showConfirmPassword ? "text" : "password"}
                                        placeholder="Confirm"
                                        data-testid="input-confirm-password-employer"
                                        {...field} 
                                      />
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        className="absolute right-0 top-0 h-full px-2 hover:bg-transparent"
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                        data-testid="button-toggle-confirm-password-employer"
                                      >
                                        {showConfirmPassword ? (
                                          <EyeOff className="h-3 w-3" />
                                        ) : (
                                          <Eye className="h-3 w-3" />
                                        )}
                                      </Button>
                                    </div>
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                          </div>

                          <FormField
                            control={employerForm.control}
                            name="companyName"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Company Name</FormLabel>
                                <FormControl>
                                  <Input placeholder="TechFlow Solutions" data-testid="input-company-name" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <div className="grid grid-cols-2 gap-2">
                            <FormField
                              control={employerForm.control}
                              name="industry"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Industry</FormLabel>
                                  <FormControl>
                                    <Input placeholder="Technology" data-testid="input-industry" {...field} />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />

                            <FormField
                              control={employerForm.control}
                              name="companySize"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Company Size</FormLabel>
                                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                                    <FormControl>
                                      <SelectTrigger data-testid="select-company-size">
                                        <SelectValue placeholder="Select size" />
                                      </SelectTrigger>
                                    </FormControl>
                                    <SelectContent>
                                      <SelectItem value="1-10">1-10 employees</SelectItem>
                                      <SelectItem value="11-50">11-50 employees</SelectItem>
                                      <SelectItem value="51-200">51-200 employees</SelectItem>
                                      <SelectItem value="201-1000">201-1000 employees</SelectItem>
                                      <SelectItem value="1000+">1000+ employees</SelectItem>
                                    </SelectContent>
                                  </Select>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                          </div>

                          <FormField
                            control={employerForm.control}
                            name="location"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Location</FormLabel>
                                <FormControl>
                                  <Input placeholder="San Francisco, CA" data-testid="input-location-employer" {...field} />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <FormField
                            control={employerForm.control}
                            name="website"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Website (Optional)</FormLabel>
                                <FormControl>
                                  <Input 
                                    type="url" 
                                    placeholder="https://www.company.com"
                                    data-testid="input-website"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />

                          <FormField
                            control={employerForm.control}
                            name="description"
                            render={({ field }) => (
                              <FormItem>
                                <FormLabel>Company Description</FormLabel>
                                <FormControl>
                                  <Textarea 
                                    placeholder="Brief description of your company, what you do, and what makes you unique..."
                                    data-testid="textarea-description"
                                    {...field} 
                                  />
                                </FormControl>
                                <FormMessage />
                              </FormItem>
                            )}
                          />
                        </div>

                        <Button type="submit" className="w-full" data-testid="button-submit-employer">
                          Create Company Account
                        </Button>
                      </form>
                    </Form>
                  </TabsContent>
                </Tabs>
              )}

              <div className="text-center text-sm text-muted-foreground mt-4">
                {isLogin ? "Don't have an account? " : "Already have an account? "}
                <button 
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-primary hover:underline"
                  data-testid="button-toggle-mode"
                >
                  {isLogin ? "Sign up" : "Sign in"}
                </button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}