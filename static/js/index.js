
// setup app and pass ui.bootstrap as dep
var myApp = angular.module("angularTypeahead", ["ui.bootstrap"]);

// define factory for data source
myApp.factory("Beers", function($http){
  
    return $http.get('beerlist');
});

// setup controller and pass data source
myApp.controller("TypeaheadCtrl", function($scope, $http, Beers) {
    $scope.selected = undefined;
     Beers.then(function(data){
         $scope.beers = data.data
     })
});