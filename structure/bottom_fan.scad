include <common.scad>



module bottom_fan_screw() {
    screw_dimeter = 2.0 + 0.5;
    translate([8, 8, 0]) cylinder(20, screw_dimeter/2, screw_dimeter/2, true);
    translate([-8, 8, 0]) cylinder(20, screw_dimeter/2, screw_dimeter/2, true);
    translate([-8, -8, 0]) cylinder(20, screw_dimeter/2, screw_dimeter/2, true);
}

module bottom_fan() {
    
    difference(){
        round_cube(20+epsilon*2, 20+epsilon*2, bottom_fan_height, 2, true);
    
        bottom_fan_screw();
    }
}

module bottom_fan_nuts() {
    nut_dimeter = 3.5 + 0.6;
    translate([8, 8, -20]) cylinder(20, nut_dimeter/2, nut_dimeter/2);
    translate([-8, 8, -20]) cylinder(20, nut_dimeter/2, nut_dimeter/2);
    translate([-8, -8, -20]) cylinder(20, nut_dimeter/2, nut_dimeter/2);

}

module bottom_fan_hole() {
    offset = 1; // round
    offset2 = 1.5; // offset to round
    
    fan_hole_diameter = 20.5;
    difference() {
        intersection() {
            cube([20, 20, 10], true);
            cylinder(10, fan_hole_diameter/2, fan_hole_diameter/2, true);
        }
        union() {
            translate([8+offset2, 8+offset2, 0]) cylinder(20, (6+offset)/2, (6+offset/2), true);
            translate([-8-offset2, 8+offset2, 0]) cylinder(20, (6+offset)/2, (6+offset/2), true);
            translate([-8-offset2, -8-offset2, 0]) cylinder(20, (6+offset)/2, (6+offset/2), true);
        }
    }
}
// bottom_fan();
// bottom_fan_nuts();
// bottom_fan_hole();