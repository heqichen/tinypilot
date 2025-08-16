include <common.scad>



/**

screw mount R = 1.25 
air out
screw hole


**/

module top_fan_outline() {
    translate([0, 0, top_fan_height/2])
    cube([top_fan_width+epsilon*2, top_fan_depth+epsilon*2, top_fan_height], true);
}



module top_fan_mount() {
    
    screw_mount_offset = 14;
    
    translate([-top_fan_width/2-epsilon, top_fan_depth/2+epsilon, 0])
    rotate([0, 0, -90])
    difference() {
        cube([3, 3, 4.5]);
        rotate([0, 0, 45])
        translate([10/2+3.5, 0, 0 ])
        cube([10, 10, 10], true);
    }
    
    
    rotate([0, 0, 270])
    translate([screw_mount_offset/2, screw_mount_offset/2, 0])
    round_cube(10, 10, 4.5, 1.5);
}



module top_fan() {
    color([1, 1, 1])
    union() {
        difference() {
            top_fan_outline();
            
            top_fan_mount();
        }
        
        // Cable place holder
        rotate([0, 0, 180+45])
        translate([0, -2.5/2, 0])
        cube([20, 2.5, top_fan_height]);
        
        // air outlet
        outlet_width = 16.5;
        posy = -top_fan_depth/2 + (top_fan_depth - outlet_width);

        translate([top_fan_width/2, posy, 0])
        cube([10, 16.5, top_fan_height]);
        
        // screw hole
        screw_diameter = 1.0 + epsilon;
        translate([8.5, -8.5, 0]) cylinder(top_fan_height, screw_diameter/2, screw_diameter/2);
        translate([-8.5, 8.5, 0]) cylinder(top_fan_height, screw_diameter/2, screw_diameter/2);


    }
}



// top_fan();

