$fn=10;
epsilon = 0.5;

module round_cube(length, width, height, dimeter) {
    r = dimeter / 2.0;
    hull() {
        translate([r, r, 0]) cylinder(height, r, r);
        translate([length-r, r, 0]) cylinder(height, r, r);
        translate([r, width-r, 0]) cylinder(height, r, r);
        translate([length-r, width-r, 0]) cylinder(height, r, r);
    }
}


module pcb_board() {
    translate([-epsilon, -epsilon, 0])
    round_cube(89 + epsilon*2, 57+epsilon*2, 1.7+epsilon, 4.0);    
}

module mounting_holes() {
    translate([23.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([23.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 4.5, 0]) cylinder(10, 3/2, 3/2, true);
    translate([81.5, 53.5, 0]) cylinder(10, 3/2, 3/2, true);
}

difference() {
    pcb_board();
    mounting_holes();
}
