import React, { useState } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger,
  DialogClose
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface MaintenanceItem {
  id: string;
  machine: string;
  sensor: string;
  timestamp: string;
  reason: string;
  status: 'Pending' | 'Acknowledged' | 'Scheduled' | 'Completed';
  scheduledDate?: string;
  assignedTo?: string;
}

interface MaintenanceLogProps {
  logs: MaintenanceItem[];
  onStatusChange?: (id: string, newStatus: MaintenanceItem['status'], additionalData?: Partial<MaintenanceItem>) => void;
}

// Helper function to format date/time
const formatTime = (timestamp: string) => {
  try {
    return new Date(timestamp).toLocaleString([], { 
      year: 'numeric', month: 'short', day: 'numeric', 
      hour: '2-digit', minute: '2-digit' 
    });
  } catch (e) {
    return timestamp;
  }
};

// Helper to get badge variant based on status
const getStatusBadgeVariant = (status: MaintenanceItem['status']): "default" | "secondary" | "destructive" | "outline" | "warning" | "info" | "success" => {
  switch (status) {
    case 'Pending': return 'warning';
    case 'Acknowledged': return 'info';
    case 'Scheduled': return 'default'; // Use default Appwrite blue/purple
    case 'Completed': return 'success';
    default: return 'secondary';
  }
};

// Add custom variants to Badge component if needed, or map to existing ones
// For demonstration, we map to existing conceptual variants
const statusVariantMapping: { [key in MaintenanceItem['status']]: string } = {
  Pending: 'bg-amber-100 text-amber-800 border-amber-200',
  Acknowledged: 'bg-blue-100 text-blue-800 border-blue-200',
  Scheduled: 'bg-indigo-100 text-indigo-800 border-indigo-200',
  Completed: 'bg-emerald-100 text-emerald-800 border-emerald-200',
};

// Schedule Dialog Component
function ScheduleDialog({ log, onStatusChange }: { log: MaintenanceItem; onStatusChange: MaintenanceLogProps['onStatusChange'] }) {
  const [scheduleDate, setScheduleDate] = useState<string>('');
  const [assignee, setAssignee] = useState<string>('');
  const [isOpen, setIsOpen] = useState(false);

  const handleSchedule = () => {
    if (scheduleDate && onStatusChange) {
      onStatusChange(log.id, 'Scheduled', { 
        scheduledDate: scheduleDate,
        assignedTo: assignee || undefined // Store as undefined if empty
      });
      setIsOpen(false); // Close dialog on successful schedule
      setScheduleDate('');
      setAssignee('');
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <span className="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 w-full">
           <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
             <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
           </svg>
          Schedule...
        </span>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Schedule Maintenance</DialogTitle>
          <DialogDescription>
            Schedule maintenance for {log.machine} - {log.sensor}.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor={`schedule-date-${log.id}`} className="text-right">
              Date & Time
            </Label>
            <Input
              id={`schedule-date-${log.id}`}
              type="datetime-local"
              className="col-span-3"
              value={scheduleDate}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setScheduleDate(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor={`assignee-${log.id}`} className="text-right">
              Assign To
            </Label>
            <Input
              id={`assignee-${log.id}`}
              placeholder="(Optional)"
              className="col-span-3"
              value={assignee}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAssignee(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <DialogClose asChild>
             <Button type="button" variant="outline">Cancel</Button>
          </DialogClose>
          <Button type="submit" onClick={handleSchedule} disabled={!scheduleDate}>
            Schedule Maintenance
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function MaintenanceLog({ logs = [], onStatusChange }: MaintenanceLogProps) {
  const [filter, setFilter] = useState<'All' | 'Pending' | 'Acknowledged' | 'Scheduled' | 'Completed'>('All');

  const filteredLogs = logs
    .filter(log => filter === 'All' || log.status === filter)
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()); // Sort by newest first

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden border border-slate-200">
      <div className="p-4 border-b border-slate-200 flex justify-between items-center">
        <h2 className="text-lg font-semibold text-slate-800">Maintenance Log</h2>
        <div className="flex space-x-1">
          {(['All', 'Pending', 'Acknowledged', 'Scheduled', 'Completed'] as const).map((status) => (
            <Button
              key={status}
              variant={filter === status ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setFilter(status)}
              className="text-xs h-8 px-2"
            >
              {status}
               <span className={`ml-1.5 inline-flex items-center justify-center px-2 py-0.5 text-xs font-medium rounded-full ${status !== 'All' ? statusVariantMapping[status] : 'bg-slate-100 text-slate-600'}`}>
                  {status === 'All' ? logs.length : logs.filter(l => l.status === status).length}
                </span>
            </Button>
          ))}
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[150px]">Machine</TableHead>
              <TableHead className="w-[150px]">Sensor/Issue</TableHead>
              <TableHead>Details</TableHead>
              <TableHead className="w-[170px]">Detected At</TableHead>
              <TableHead className="w-[120px]">Status</TableHead>
              <TableHead className="w-[170px]">Scheduled For</TableHead>
              <TableHead className="w-[120px]">Assignee</TableHead>
              <TableHead className="text-right w-[100px]">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredLogs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="h-24 text-center text-slate-500">
                  No maintenance logs match the current filter.
                </TableCell>
              </TableRow>
            ) : (
              filteredLogs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="font-medium">{log.machine}</TableCell>
                  <TableCell>{log.sensor}</TableCell>
                  <TableCell className="text-sm text-slate-600 max-w-xs truncate" title={log.reason}>{log.reason}</TableCell>
                  <TableCell className="text-xs">{formatTime(log.timestamp)}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={`text-xs ${statusVariantMapping[log.status]}`}>
                      {log.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs">{log.scheduledDate ? formatTime(log.scheduledDate) : '-'}</TableCell>
                  <TableCell className="text-xs">{log.assignedTo || '-'}</TableCell>
                  <TableCell className="text-right">
                    {onStatusChange && (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7">
                            <span className="sr-only">Open menu</span>
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                               <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                            </svg>
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          {log.status === 'Pending' && (
                            <DropdownMenuItem onClick={() => onStatusChange(log.id, 'Acknowledged')}>
                              Acknowledge
                            </DropdownMenuItem>
                          )}
                          {(log.status === 'Pending' || log.status === 'Acknowledged') && (
                            <ScheduleDialog log={log} onStatusChange={onStatusChange} />
                          )}
                          {log.status === 'Scheduled' && (
                            <DropdownMenuItem onClick={() => onStatusChange(log.id, 'Completed')}>
                              Mark as Completed
                            </DropdownMenuItem>
                          )}
                          {(log.status === 'Acknowledged' || log.status === 'Scheduled' ) && log.status !== 'Completed' && (
                             <DropdownMenuItem onClick={() => onStatusChange(log.id, 'Pending')} className="text-slate-600">
                              Reset to Pending
                            </DropdownMenuItem>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
} 