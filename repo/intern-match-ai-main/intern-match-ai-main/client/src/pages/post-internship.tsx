import { useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Badge } from "@/components/ui/badge";
import { X, ArrowLeft } from "lucide-react";
import { createInternship } from "@/lib/api";

const internshipSchema = z.object({
  title: z.string().min(5, "Title must be at least 5 characters"),
  description: z.string().min(20, "Description must be at least 20 characters"),
  location: z.string().min(2, "Location is required"),
  duration: z.string().min(1, "Duration is required"),
  stipend: z.string().optional(),
  requirements: z.string().min(10, "Requirements must be at least 10 characters"),
  skills: z.string().min(5, "Please list required skills"),
  applicationDeadline: z.string().min(1, "Application deadline is required"),
  startDate: z.string().min(1, "Start date is required"),
  maxApplications: z.number().min(1, "Max applications must be at least 1"),
  status: z.enum(["open", "closed"]).default("open"),
});

export default function PostInternshipPage() {
  const [, setLocation] = useLocation();
  const [skillsList, setSkillsList] = useState<string[]>([]);
  const [currentSkill, setCurrentSkill] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const form = useForm({
    resolver: zodResolver(internshipSchema),
    defaultValues: {
      title: "",
      description: "",
      location: "",
      duration: "",
      stipend: "",
      requirements: "",
      skills: "",
      applicationDeadline: "",
      startDate: "",
      maxApplications: 20,
      status: "open" as const,
    },
  });

  const addSkill = () => {
    if (currentSkill.trim() && !skillsList.includes(currentSkill.trim())) {
      const newSkills = [...skillsList, currentSkill.trim()];
      setSkillsList(newSkills);
      form.setValue("skills", newSkills.join(", "));
      setCurrentSkill("");
    }
  };

  const removeSkill = (skillToRemove: string) => {
    const newSkills = skillsList.filter(skill => skill !== skillToRemove);
    setSkillsList(newSkills);
    form.setValue("skills", newSkills.join(", "));
  };

  const onSubmit = async (values: z.infer<typeof internshipSchema>) => {
    try {
      setIsSubmitting(true);
      await createInternship({
        ...values,
        skills: skillsList,
      });
      setLocation("/employer");
    } catch (error) {
      console.error("Failed to create internship:", error);
      form.setError("title", { message: "Failed to create internship. Please try again." });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <Button 
            variant="ghost" 
            onClick={() => setLocation("/employer")}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
          <h1 className="text-3xl font-bold">Post New Internship</h1>
          <p className="text-muted-foreground">Create a new internship opportunity</p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Internship Details</CardTitle>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="title"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Internship Title</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. Software Engineer Intern" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="location"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Location</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. San Francisco, CA" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="description"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Description</FormLabel>
                      <FormControl>
                        <Textarea 
                          placeholder="Describe the internship role, responsibilities, and what the intern will learn..."
                          className="min-h-[120px]"
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="requirements"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Requirements</FormLabel>
                      <FormControl>
                        <Textarea 
                          placeholder="List the required qualifications, experience, education level, etc..."
                          className="min-h-[100px]"
                          {...field} 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="duration"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Duration</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder="Select duration" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="1 month">1 month</SelectItem>
                            <SelectItem value="2 months">2 months</SelectItem>
                            <SelectItem value="3 months">3 months</SelectItem>
                            <SelectItem value="4 months">4 months</SelectItem>
                            <SelectItem value="6 months">6 months</SelectItem>
                            <SelectItem value="12 months">12 months</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="stipend"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Stipend (Optional)</FormLabel>
                        <FormControl>
                          <Input placeholder="e.g. $3,000/month" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                {/* Skills */}
                <div className="space-y-4">
                  <FormLabel>Required Skills</FormLabel>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a skill..."
                      value={currentSkill}
                      onChange={(e) => setCurrentSkill(e.target.value)}
                      onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addSkill())}
                    />
                    <Button type="button" onClick={addSkill}>Add</Button>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {skillsList.map((skill) => (
                      <Badge key={skill} variant="secondary" className="flex items-center gap-1">
                        {skill}
                        <X 
                          className="h-3 w-3 cursor-pointer" 
                          onClick={() => removeSkill(skill)}
                        />
                      </Badge>
                    ))}
                  </div>
                  <FormField
                    control={form.control}
                    name="skills"
                    render={() => (
                      <FormItem>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <FormField
                    control={form.control}
                    name="startDate"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Start Date</FormLabel>
                        <FormControl>
                          <Input type="date" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="applicationDeadline"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Application Deadline</FormLabel>
                        <FormControl>
                          <Input type="date" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="maxApplications"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Max Applications</FormLabel>
                        <FormControl>
                          <Input 
                            type="number" 
                            min="1"
                            max="1000"
                            {...field}
                            onChange={(e) => field.onChange(parseInt(e.target.value))}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="flex gap-4 pt-6">
                  <Button type="submit" disabled={isSubmitting}>
                    {isSubmitting ? "Creating..." : "Post Internship"}
                  </Button>
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={() => setLocation("/employer")}
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            </Form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
