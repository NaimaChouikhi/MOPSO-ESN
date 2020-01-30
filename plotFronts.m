pop_costs1=[pop(F{1}).Cost];
pop_costs2=[pop(F{2}).Cost];
pop_costs3=[pop(F{3}).Cost];
plot3(pop_costs1(1,:),pop_costs1(2,:),pop_costs1(3,:),'b*');
 hold on;
plot3(pop_costs2(1,:),pop_costs2(2,:),pop_costs2(3,:),'r*');
 hold on;
plot3(pop_costs3(1,:),pop_costs3(2,:),pop_costs3(3,:),'g*');
 xlabel('1^{st} Objective: Reservoir size ');
    ylabel('2^{nd} Objective: Reservoir connectivity rate');
    zlabel('3^{rd} Objective: Accuracy(RMSE) ');
    grid on;
    
    hold off;