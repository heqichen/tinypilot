include <common.scad>
include <top_fan.scad> 
include <mainboard.scad>
include <bottom_fan.scad>
module top_shell(){include<top_shell.scad>}
module bottom_shell(){include <bottom_shell.scad>}

layer_offset = 15;

translate([0, 0, layer_offset*4.5])
top_shell();

translate([0, 0, layer_offset*3])
translate([cpu_pos_x + top_fan_offset_x, cpu_pos_y-2, pcb_height+top_space_height])
rotate([0, 180, 0]) top_fan();


translate([0, 0, layer_offset*2])
mock_mainboard();


translate([0, 0, layer_offset*1.5])
translate([bottom_fan_center_x, bottom_fan_center_y, -bottom_space_height]) bottom_fan();


bottom_shell();